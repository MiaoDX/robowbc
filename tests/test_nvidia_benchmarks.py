import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "artifacts/benchmarks/nvidia/cases.json"
NORMALIZER_PATH = ROOT / "scripts/normalize_nvidia_benchmarks.py"

SPEC = importlib.util.spec_from_file_location("normalize_nvidia_benchmarks", NORMALIZER_PATH)
NORMALIZER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(NORMALIZER)


class NvidiaBenchmarkTests(unittest.TestCase):
    def make_stub_decoupled_repo(self, root: Path) -> Path:
        repo_dir = root / "GR00T-WholeBodyControl"
        policy_dir = repo_dir / "decoupled_wbc" / "control" / "policy"
        resource_dir = repo_dir / "decoupled_wbc" / "sim2mujoco" / "resources" / "robots" / "g1"
        policy_dir.mkdir(parents=True)
        resource_dir.mkdir(parents=True)

        for package_dir in (
            repo_dir / "decoupled_wbc",
            repo_dir / "decoupled_wbc" / "control",
            policy_dir,
        ):
            (package_dir / "__init__.py").write_text("", encoding="utf-8")

        (resource_dir / "g1_gear_wbc.yaml").write_text("dummy: true\n", encoding="utf-8")
        (policy_dir / "g1_gear_wbc_policy.py").write_text(
            textwrap.dedent(
                """
                import collections
                import numpy as np


                class G1GearWbcPolicy:
                    def __init__(self, robot_model, config, model_path):
                        self.robot_model = robot_model
                        self.config = {
                            "obs_history_len": 6,
                            "num_obs": 516,
                            "num_actions": 15,
                            "default_angles": np.zeros(15, dtype=np.float32),
                            "cmd_init": np.zeros(3, dtype=np.float32),
                            "height_cmd": 0.74,
                            "freq_cmd": 1.5,
                        }
                        self.obs_history = collections.deque(maxlen=self.config["obs_history_len"])
                        self.obs_buffer = np.zeros(self.config["num_obs"], dtype=np.float32)
                        self.counter = 0
                        self.action = np.zeros(self.config["num_actions"], dtype=np.float32)
                        self.target_dof_pos = self.config["default_angles"].copy()
                        self.cmd = self.config["cmd_init"].copy()
                        self.height_cmd = float(self.config["height_cmd"])
                        self.freq_cmd = float(self.config["freq_cmd"])
                        self.roll_cmd = 0.0
                        self.pitch_cmd = 0.0
                        self.yaw_cmd = 0.0
                        self.gait_indices = np.zeros((1,), dtype=np.float32)
                        self.obs_tensor = None
                        self.use_policy_action = False
                        self.use_teleop_policy_cmd = False
                        self.observation = None

                    def set_use_teleop_policy_cmd(self, enabled):
                        self.use_teleop_policy_cmd = enabled

                    def set_observation(self, observation):
                        self.observation = observation
                        body_indices = self.robot_model.get_joint_group_indices("body")
                        q = observation["q"][body_indices]
                        single_obs = np.zeros(86, dtype=np.float32)
                        single_obs[0:3] = self.cmd[:3]
                        single_obs[13 : 13 + len(q)] = q
                        self.obs_history.append(single_obs)
                        while len(self.obs_history) < self.config["obs_history_len"]:
                            self.obs_history.appendleft(np.zeros_like(single_obs))
                        for index, hist_obs in enumerate(self.obs_history):
                            start = index * len(single_obs)
                            end = start + len(single_obs)
                            self.obs_buffer[start:end] = hist_obs
                        self.obs_tensor = self.obs_buffer.reshape(1, -1)

                    def get_action(self, time=None):
                        del time
                        fill = 1.0 if np.linalg.norm(self.cmd) >= 0.05 else -1.0
                        self.action = np.full(self.config["num_actions"], fill, dtype=np.float32)
                        zeros = np.zeros(self.config["num_actions"], dtype=np.float32)
                        return {"body_action": (self.action.copy(), zeros, zeros)}
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        subprocess.run(["git", "-c", "init.defaultBranch=main", "init"], check=True, cwd=repo_dir)
        subprocess.run(["git", "add", "."], check=True, cwd=repo_dir)
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Test User",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-m",
                "init stub repo",
            ],
            check=True,
            cwd=repo_dir,
        )
        return repo_dir

    def make_decoupled_model_dir(self, root: Path) -> Path:
        model_dir = root / "decoupled-models"
        model_dir.mkdir(parents=True)
        fixture_dir = ROOT / "crates/robowbc-ort/tests/fixtures"
        shutil.copyfile(
            fixture_dir / "test_constant_balance.onnx",
            model_dir / "GR00T-WholeBodyControl-Balance.onnx",
        )
        shutil.copyfile(
            fixture_dir / "test_constant_walk.onnx",
            model_dir / "GR00T-WholeBodyControl-Walk.onnx",
        )
        return model_dir

    def test_registry_validates_and_contains_expected_cases(self) -> None:
        registry = NORMALIZER.load_registry(REGISTRY_PATH)
        case_ids = {case["case_id"] for case in registry["cases"]}
        self.assertEqual(
            case_ids,
            {
                "gear_sonic_velocity/cold_start_tick",
                "gear_sonic_velocity/warm_steady_state_tick",
                "gear_sonic_velocity/replan_tick",
                "gear_sonic_tracking/standing_placeholder_tick",
                "decoupled_wbc/walk_predict",
                "decoupled_wbc/balance_predict",
                "gear_sonic/end_to_end_cli_loop",
                "decoupled_wbc/end_to_end_cli_loop",
            },
        )

    def test_normalize_criterion_samples(self) -> None:
        registry = NORMALIZER.load_registry(REGISTRY_PATH)
        case = NORMALIZER.registry_case(registry, "decoupled_wbc/walk_predict")
        with tempfile.TemporaryDirectory() as tmpdir:
            criterion_root = Path(tmpdir) / "criterion"
            bench_dir = criterion_root / "policy_decoupled_wbc" / "walk_predict" / "new"
            bench_dir.mkdir(parents=True)

            (bench_dir / "benchmark.json").write_text(
                json.dumps(
                    {
                        "full_id": case["criterion_id"],
                        "group_id": "policy/decoupled_wbc",
                        "function_id": "walk_predict",
                    }
                ),
                encoding="utf-8",
            )
            (bench_dir / "sample.json").write_text(
                json.dumps(
                    {
                        "iters": [1, 2, 4],
                        "times": [100, 240, 520],
                    }
                ),
                encoding="utf-8",
            )

            samples_ns, raw_source = NORMALIZER.criterion_samples_ns(
                criterion_root, case["criterion_id"]
            )
            self.assertEqual(samples_ns, [100.0, 120.0, 130.0])
            self.assertTrue(raw_source.endswith("sample.json"))

            artifact = NORMALIZER.build_artifact(
                case=case,
                stack="robowbc",
                upstream_commit="upstream",
                robowbc_commit="robowbc",
                provider="cpu",
                host_fingerprint="host",
                samples_ns=samples_ns,
                hz=None,
                notes="criterion test",
                source_command="cargo bench ...",
                raw_source=raw_source,
                status="ok",
            )
            self.assertEqual(artifact["p50_ns"], 120)
            self.assertEqual(artifact["p95_ns"], 129)
            self.assertEqual(artifact["p99_ns"], 130)

    def test_normalize_run_report(self) -> None:
        registry = NORMALIZER.load_registry(REGISTRY_PATH)
        case = NORMALIZER.registry_case(registry, "gear_sonic/end_to_end_cli_loop")
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "metrics": {"achieved_frequency_hz": 32.3},
                        "frames": [
                            {"inference_latency_ms": 1.0},
                            {"inference_latency_ms": 1.5},
                            {"inference_latency_ms": 2.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            samples_ns, hz, raw_source = NORMALIZER.run_report_samples_ns(report_path)
            artifact = NORMALIZER.build_artifact(
                case=case,
                stack="robowbc",
                upstream_commit="upstream",
                robowbc_commit="robowbc",
                provider="cpu",
                host_fingerprint="host",
                samples_ns=samples_ns,
                hz=hz,
                notes="run report test",
                source_command="cargo run ...",
                raw_source=raw_source,
                status="ok",
            )
            self.assertEqual(artifact["p50_ns"], 1_500_000)
            self.assertEqual(artifact["hz"], 32.3)

    def test_official_wrapper_blocks_decoupled_case_when_models_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = self.make_stub_decoupled_repo(Path(tmpdir))
            env = os.environ.copy()
            env["DECOUPLED_WBC_MODEL_DIR"] = str(Path(tmpdir) / "missing-decoupled-models")
            subprocess.run(
                [
                    "bash",
                    str(ROOT / "scripts/bench_nvidia_official.sh"),
                    "--case",
                    "decoupled_wbc/walk_predict",
                    "--repo-dir",
                    str(repo_dir),
                    "--output-root",
                    tmpdir,
                ],
                check=True,
                cwd=ROOT,
                env=env,
            )
            artifact_path = Path(tmpdir) / "decoupled_wbc__walk_predict.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            self.assertEqual(artifact["case_id"], "decoupled_wbc/walk_predict")
            self.assertEqual(artifact["status"], "blocked")
            self.assertEqual(artifact["stack"], "official_nvidia")
            self.assertIn("scripts/download_decoupled_wbc_models.sh", artifact["notes"])

    def test_official_wrapper_runs_decoupled_cases_and_blocks_gear_sonic_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            repo_dir = self.make_stub_decoupled_repo(tmp_path)
            model_dir = self.make_decoupled_model_dir(tmp_path)
            output_dir = tmp_path / "normalized"
            env = os.environ.copy()
            env["DECOUPLED_WBC_MODEL_DIR"] = str(model_dir)

            subprocess.run(
                [
                    "bash",
                    str(ROOT / "scripts/bench_nvidia_official.sh"),
                    "--all",
                    "--repo-dir",
                    str(repo_dir),
                    "--output-root",
                    str(output_dir),
                    "--samples",
                    "3",
                    "--ticks",
                    "5",
                ],
                check=True,
                cwd=ROOT,
                env=env,
            )

            registry = NORMALIZER.load_registry(REGISTRY_PATH)
            for case in registry["cases"]:
                artifact_path = output_dir / f"{case['case_id'].replace('/', '__')}.json"
                self.assertTrue(artifact_path.is_file(), msg=f"missing {artifact_path}")
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                self.assertEqual(artifact["case_id"], case["case_id"])
                self.assertEqual(artifact["stack"], "official_nvidia")

                if case["case_id"].startswith("decoupled_wbc/"):
                    self.assertEqual(artifact["status"], "ok")
                    self.assertGreaterEqual(artifact["samples"], 3)
                    self.assertTrue(Path(artifact["raw_source"]).is_file())
                    if case["case_id"] == "decoupled_wbc/end_to_end_cli_loop":
                        self.assertIsNotNone(artifact["hz"])
                    else:
                        self.assertIsNone(artifact["hz"])
                else:
                    self.assertEqual(artifact["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
