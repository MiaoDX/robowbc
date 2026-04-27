import argparse
import importlib.util
import json
import os
import stat
import shutil
import subprocess
import tempfile
import textwrap
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "benchmarks/nvidia/cases.json"
NORMALIZER_PATH = ROOT / "scripts/normalize_nvidia_benchmarks.py"
RENDER_PATH = ROOT / "scripts/render_nvidia_benchmark_summary.py"
ROBOHARNESS_REPORT_PATH = ROOT / "scripts/roboharness_report.py"
ROBOWBC_COMPARE_PATH = ROOT / "scripts/bench_robowbc_compare.py"
OFFICIAL_COMPARE_PATH = ROOT / "scripts/bench_nvidia_official.py"

SPEC = importlib.util.spec_from_file_location("normalize_nvidia_benchmarks", NORMALIZER_PATH)
NORMALIZER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(NORMALIZER)

RENDER_SPEC = importlib.util.spec_from_file_location("render_nvidia_benchmark_summary", RENDER_PATH)
RENDER = importlib.util.module_from_spec(RENDER_SPEC)
assert RENDER_SPEC.loader is not None
RENDER_SPEC.loader.exec_module(RENDER)

ROBOHARNESS_SPEC = importlib.util.spec_from_file_location(
    "roboharness_report", ROBOHARNESS_REPORT_PATH
)
ROBOHARNESS = importlib.util.module_from_spec(ROBOHARNESS_SPEC)
assert ROBOHARNESS_SPEC.loader is not None
ROBOHARNESS_SPEC.loader.exec_module(ROBOHARNESS)

ROBO_COMPARE_SPEC = importlib.util.spec_from_file_location(
    "bench_robowbc_compare", ROBOWBC_COMPARE_PATH
)
ROBO_COMPARE = importlib.util.module_from_spec(ROBO_COMPARE_SPEC)
assert ROBO_COMPARE_SPEC.loader is not None
ROBO_COMPARE_SPEC.loader.exec_module(ROBO_COMPARE)

OFFICIAL_COMPARE_SPEC = importlib.util.spec_from_file_location(
    "bench_nvidia_official", OFFICIAL_COMPARE_PATH
)
OFFICIAL_COMPARE = importlib.util.module_from_spec(OFFICIAL_COMPARE_SPEC)
assert OFFICIAL_COMPARE_SPEC.loader is not None
OFFICIAL_COMPARE_SPEC.loader.exec_module(OFFICIAL_COMPARE)


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

    def make_gear_sonic_model_dir(self, root: Path) -> Path:
        model_dir = root / "gear-sonic-models"
        model_dir.mkdir(parents=True)
        for filename in ("model_encoder.onnx", "model_decoder.onnx", "planner_sonic.onnx"):
            (model_dir / filename).write_text("stub\n", encoding="utf-8")
        return model_dir

    def make_fake_cargo(self, root: Path) -> Path:
        fake_bin = root / "fake-bin"
        fake_bin.mkdir(parents=True)
        cargo_path = fake_bin / "cargo"
        cargo_path.write_text(
            textwrap.dedent(
                """
                #!/usr/bin/env python3
                import json
                import os
                import pathlib
                import re
                import sys


                def main() -> int:
                    args = sys.argv[1:]
                    if not args or args[0] != "run" or "--config" not in args:
                        raise SystemExit(f"unexpected cargo invocation: {args!r}")

                    config_path = pathlib.Path(args[args.index("--config") + 1])
                    config_text = config_path.read_text(encoding="utf-8")
                    mujoco_download_dir = os.environ.get("MUJOCO_DOWNLOAD_DIR")
                    if not mujoco_download_dir:
                        raise SystemExit("MUJOCO_DOWNLOAD_DIR was not provided to cargo run")
                    if not pathlib.Path(mujoco_download_dir).is_absolute():
                        raise SystemExit("MUJOCO_DOWNLOAD_DIR must be absolute")
                    match = re.search(r'^output_path\\s*=\\s*"([^"]+)"\\s*$', config_text, re.MULTILINE)
                    if match is None:
                        raise SystemExit("report output_path missing from generated config")

                    report_path = pathlib.Path(match.group(1))
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(
                        json.dumps(
                            {
                                "metrics": {"achieved_frequency_hz": 41.25},
                                "frames": [
                                    {"inference_latency_ms": 1.2},
                                    {"inference_latency_ms": 1.4},
                                    {"inference_latency_ms": 1.6},
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )
                    return 0


                if __name__ == "__main__":
                    raise SystemExit(main())
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        cargo_path.chmod(cargo_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return fake_bin

    def make_failing_bench_cargo(self, root: Path) -> Path:
        fake_bin = root / "fake-bench-bin"
        fake_bin.mkdir(parents=True)
        cargo_path = fake_bin / "cargo"
        cargo_path.write_text(
            textwrap.dedent(
                """
                #!/usr/bin/env python3
                import os
                import sys


                def main() -> int:
                    args = sys.argv[1:]
                    if not args or args[0] != "bench":
                        raise SystemExit(f"unexpected cargo invocation: {args!r}")
                    provider = os.environ.get("ROBOWBC_BENCH_PROVIDER", "<missing>")
                    print(
                        f"provider init failed for {provider}: missing runtime dependency libcuda.so.1",
                        file=sys.stderr,
                    )
                    return 1


                if __name__ == "__main__":
                    raise SystemExit(main())
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        cargo_path.chmod(cargo_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return fake_bin

    def make_fake_robowbc_binary(self, root: Path) -> Path:
        fake_bin = root / "fake-robowbc"
        fake_bin.write_text(
            textwrap.dedent(
                """
                #!/usr/bin/env python3
                import json
                import pathlib
                import re
                import sys


                def main() -> int:
                    args = sys.argv[1:]
                    if args[:1] != ["run"] or "--config" not in args:
                        raise SystemExit(f"unexpected robowbc invocation: {args!r}")

                    config_path = pathlib.Path(args[args.index("--config") + 1])
                    config_text = config_path.read_text(encoding="utf-8")
                    output_match = re.search(r'^output_path\\s*=\\s*"([^"]+)"\\s*$', config_text, re.MULTILINE)
                    if output_match is None:
                        raise SystemExit("missing report output_path")

                    output_path = pathlib.Path(output_match.group(1))
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(
                        json.dumps(
                            {
                                "policy_name": "gear_sonic",
                                "robot_name": "unitree_g1",
                                "command_kind": "velocity",
                                "command_data": [0.3, 0.0, 0.0],
                                "metrics": {
                                    "ticks": 3,
                                    "average_inference_ms": 1.25,
                                    "achieved_frequency_hz": 48.5,
                                    "dropped_frames": 0,
                                },
                                "frames": [
                                    {"tick": 0, "actual_positions": [0.0], "inference_latency_ms": 1.1},
                                    {"tick": 1, "actual_positions": [0.1], "inference_latency_ms": 1.2},
                                    {"tick": 2, "actual_positions": [0.2], "inference_latency_ms": 1.45},
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )
                    return 0


                if __name__ == "__main__":
                    raise SystemExit(main())
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return fake_bin

    def make_fake_gear_sonic_harness(self, root: Path) -> Path:
        harness_path = root / "fake-gear-sonic-harness"
        harness_path.write_text(
            textwrap.dedent(
                """
                #!/usr/bin/env python3
                import sys


                def main() -> int:
                    args = sys.argv[1:]
                    if "--provider" not in args:
                        raise SystemExit("missing --provider")
                    provider = args[args.index("--provider") + 1]
                    print(
                        f"requested provider {provider} failed: CUDAExecutionProvider not available",
                        file=sys.stderr,
                    )
                    return 1


                if __name__ == "__main__":
                    raise SystemExit(main())
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        harness_path.chmod(
            harness_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )
        return harness_path

    def test_registry_validates_and_contains_expected_cases(self) -> None:
        registry = NORMALIZER.load_registry(REGISTRY_PATH)
        case_ids = {case["case_id"] for case in registry["cases"]}
        self.assertEqual(
            case_ids,
            {
                "gear_sonic/planner_only_cold_start",
                "gear_sonic/planner_only_steady_state",
                "gear_sonic/encoder_decoder_only_tracking_tick",
                "gear_sonic/full_velocity_tick_cold_start",
                "gear_sonic/full_velocity_tick_steady_state",
                "gear_sonic/full_velocity_tick_replan_boundary",
                "decoupled_wbc/walk_predict",
                "decoupled_wbc/balance_predict",
                "gear_sonic/end_to_end_cli_loop",
                "decoupled_wbc/end_to_end_cli_loop",
            },
        )

    def test_wrappers_list_cases_match_the_registry(self) -> None:
        registry = NORMALIZER.load_registry(REGISTRY_PATH)
        expected = [case["case_id"] for case in registry["cases"]]

        for script_path in (ROBOWBC_COMPARE_PATH, OFFICIAL_COMPARE_PATH):
            result = subprocess.run(
                ["python3", str(script_path), "--list-cases"],
                check=True,
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.stdout.strip().splitlines(), expected)

    def test_robowbc_source_command_for_case_includes_provider(self) -> None:
        args = argparse.Namespace(
            output_root=Path("/tmp/robowbc-bench-out"),
            provider="cuda",
        )
        command = ROBO_COMPARE.source_command_for_case(
            args, "gear_sonic/full_velocity_tick_steady_state"
        )
        self.assertEqual(
            command,
            "python3 scripts/bench_robowbc_compare.py --case "
            "gear_sonic/full_velocity_tick_steady_state --provider cuda "
            "--output-root /tmp/robowbc-bench-out",
        )

    def test_official_source_command_for_case_includes_provider_and_overrides(self) -> None:
        args = argparse.Namespace(
            provider="tensor_rt",
            repo_dir=Path("/tmp/official-src"),
            output_root=Path("/tmp/official-bench-out"),
            samples=9,
            ticks=17,
            control_frequency_hz=60,
        )
        command = OFFICIAL_COMPARE.source_command_for_case(
            args, "gear_sonic/planner_only_cold_start"
        )
        self.assertEqual(
            command,
            "python3 scripts/bench_nvidia_official.py --case "
            "gear_sonic/planner_only_cold_start --provider tensor_rt "
            "--repo-dir /tmp/official-src --output-root /tmp/official-bench-out "
            "--samples 9 --ticks 17 --control-frequency-hz 60",
        )

    def test_rewrite_gear_sonic_provider_blocks_updates_all_sections(self) -> None:
        config_text = (ROOT / "configs/sonic_g1.toml").read_text(encoding="utf-8")

        for provider, expected in (
            ("cpu", {"type": "cpu"}),
            ("cuda", {"type": "cuda", "device_id": 0}),
            ("tensor_rt", {"type": "tensor_rt", "device_id": 0}),
        ):
            with self.subTest(provider=provider):
                rewritten = ROBO_COMPARE.rewrite_gear_sonic_provider_blocks(
                    config_text, provider
                )
                parsed = tomllib.loads(rewritten)
                self.assertEqual(parsed["policy"]["config"]["encoder"]["execution_provider"], expected)
                self.assertEqual(parsed["policy"]["config"]["decoder"]["execution_provider"], expected)
                self.assertEqual(parsed["policy"]["config"]["planner"]["execution_provider"], expected)

    def test_compose_benchmark_cli_config_rewrites_providers_and_appends_report(self) -> None:
        config_text = (ROOT / "configs/sonic_g1.toml").read_text(encoding="utf-8")
        report_path = Path("/tmp/gear-sonic-bench-report.json")

        composed = ROBO_COMPARE.compose_benchmark_cli_config(
            config_text,
            provider="cuda",
            report_path=report_path,
            rewrite_gear_sonic_providers=True,
        )

        parsed = tomllib.loads(composed)
        expected_provider = {"type": "cuda", "device_id": 0}
        self.assertEqual(
            parsed["policy"]["config"]["encoder"]["execution_provider"], expected_provider
        )
        self.assertEqual(
            parsed["policy"]["config"]["decoder"]["execution_provider"], expected_provider
        )
        self.assertEqual(
            parsed["policy"]["config"]["planner"]["execution_provider"], expected_provider
        )
        self.assertEqual(parsed["report"]["output_path"], str(report_path))
        self.assertEqual(parsed["report"]["max_frames"], 200)

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

    def test_render_html_summary_contains_case_links(self) -> None:
        registry = NORMALIZER.load_registry(REGISTRY_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "official").mkdir()
            (root / "robowbc").mkdir()

            case = NORMALIZER.registry_case(registry, "decoupled_wbc/walk_predict")
            for stack_name, output_dir, p50_ns in (
                ("official_nvidia", root / "official", 170_000),
                ("robowbc", root / "robowbc", 41_000),
            ):
                artifact = NORMALIZER.build_artifact(
                    case=case,
                    stack=stack_name,
                    upstream_commit="bc38f6d0ce6cab4589e025037ad0bfbab7ba73d8",
                    robowbc_commit="cab3f22490f43c4a366a9f4cf769a250bcbe4063",
                    provider="cpu",
                    host_fingerprint="ci-host",
                    samples_ns=[p50_ns, p50_ns + 1_000, p50_ns + 2_000],
                    hz=None,
                    notes="html test",
                    source_command="python3 script.py",
                    raw_source="raw.json",
                    status="ok",
                )
                output_path = output_dir / "decoupled_wbc__walk_predict.json"
                output_path.write_text(json.dumps(artifact), encoding="utf-8")

            summary = RENDER.build_summary(registry, root)
            html_summary = RENDER.render_html(summary)

            self.assertIn("<!DOCTYPE html>", html_summary)
            self.assertIn("RoboWBC NVIDIA Comparison", html_summary)
            self.assertIn("decoupled_wbc/walk_predict", html_summary)
            self.assertIn("robowbc/decoupled_wbc__walk_predict.json", html_summary)
            self.assertIn("official/decoupled_wbc__walk_predict.json", html_summary)
            self.assertIn("Path group", html_summary)
            self.assertIn("Decoupled WBC", html_summary)
            self.assertIn("Site home", html_summary)
            self.assertIn("Markdown summary", html_summary)
            self.assertIn("Case registry", html_summary)

    def test_official_wrapper_blocks_decoupled_case_when_models_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = self.make_stub_decoupled_repo(Path(tmpdir))
            env = os.environ.copy()
            env["DECOUPLED_WBC_MODEL_DIR"] = str(Path(tmpdir) / "missing-decoupled-models")
            subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts/bench_nvidia_official.py"),
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

    def test_official_wrapper_blocks_decoupled_non_cpu_provider_without_relabeling(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = self.make_stub_decoupled_repo(Path(tmpdir))
            output_dir = Path(tmpdir) / "official-artifacts"

            subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts/bench_nvidia_official.py"),
                    "--case",
                    "decoupled_wbc/walk_predict",
                    "--provider",
                    "cuda",
                    "--repo-dir",
                    str(repo_dir),
                    "--output-root",
                    str(output_dir),
                ],
                check=True,
                cwd=ROOT,
            )

            artifact = json.loads(
                (output_dir / "decoupled_wbc__walk_predict.json").read_text(encoding="utf-8")
            )
            self.assertEqual(artifact["status"], "blocked")
            self.assertEqual(artifact["provider"], "cuda")
            self.assertIn("stays CPU-only in this phase", artifact["notes"])
            self.assertIn("quietly relabeling a CPU measurement", artifact["notes"])

    def test_official_wrapper_runs_decoupled_cases_and_blocks_gear_sonic_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            repo_dir = self.make_stub_decoupled_repo(tmp_path)
            model_dir = self.make_decoupled_model_dir(tmp_path)
            output_dir = tmp_path / "normalized"
            env = os.environ.copy()
            env["DECOUPLED_WBC_MODEL_DIR"] = str(model_dir)
            env["GEAR_SONIC_MODEL_DIR"] = str(tmp_path / "missing-gear-models")

            subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts/bench_nvidia_official.py"),
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

            policy_dir = (
                repo_dir
                / "decoupled_wbc"
                / "sim2mujoco"
                / "resources"
                / "robots"
                / "g1"
                / "policy"
            )
            self.assertFalse(policy_dir.exists(), msg="official harness should not mutate the repo")

    def test_robowbc_cli_wrapper_records_actual_case_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            fake_bin = self.make_fake_cargo(tmp_path)
            gear_sonic_model_dir = self.make_gear_sonic_model_dir(tmp_path)
            decoupled_model_dir = self.make_decoupled_model_dir(tmp_path)
            output_dir = tmp_path / "robowbc-artifacts"
            env = os.environ.copy()
            env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
            env["GEAR_SONIC_MODEL_DIR"] = str(gear_sonic_model_dir)
            env["DECOUPLED_WBC_MODEL_DIR"] = str(decoupled_model_dir)

            for case_id in (
                "gear_sonic/end_to_end_cli_loop",
                "decoupled_wbc/end_to_end_cli_loop",
            ):
                subprocess.run(
                    [
                        "python3",
                        str(ROOT / "scripts/bench_robowbc_compare.py"),
                        "--case",
                        case_id,
                        "--output-root",
                        str(output_dir),
                    ],
                    check=True,
                    cwd=ROOT,
                    env=env,
                )

                artifact_path = output_dir / f"{case_id.replace('/', '__')}.json"
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                self.assertEqual(artifact["status"], "ok")
                self.assertEqual(
                    artifact["source_command"],
                    ROBO_COMPARE.source_command_for_case(
                        argparse.Namespace(output_root=output_dir, provider="cpu"),
                        case_id,
                    ),
                )
                self.assertAlmostEqual(artifact["hz"], 41.25)

    def test_robowbc_wrapper_blocks_decoupled_non_cpu_provider_without_relabeling(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "robowbc-artifacts"
            subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts/bench_robowbc_compare.py"),
                    "--case",
                    "decoupled_wbc/walk_predict",
                    "--provider",
                    "cuda",
                    "--output-root",
                    str(output_dir),
                ],
                check=True,
                cwd=ROOT,
            )

            artifact = json.loads(
                (output_dir / "decoupled_wbc__walk_predict.json").read_text(encoding="utf-8")
            )
            self.assertEqual(artifact["status"], "blocked")
            self.assertEqual(artifact["provider"], "cuda")
            self.assertIn("stays CPU-only in this phase", artifact["notes"])
            self.assertIn("quietly relabeling a CPU measurement", artifact["notes"])

    def test_robowbc_microbench_provider_failure_emits_blocked_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            fake_bin = self.make_failing_bench_cargo(tmp_path)
            gear_sonic_model_dir = self.make_gear_sonic_model_dir(tmp_path)
            output_dir = tmp_path / "robowbc-artifacts"
            env = os.environ.copy()
            env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
            env["GEAR_SONIC_MODEL_DIR"] = str(gear_sonic_model_dir)

            subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts/bench_robowbc_compare.py"),
                    "--case",
                    "gear_sonic/planner_only_cold_start",
                    "--provider",
                    "cuda",
                    "--output-root",
                    str(output_dir),
                ],
                check=True,
                cwd=ROOT,
                env=env,
            )

            artifact = json.loads(
                (output_dir / "gear_sonic__planner_only_cold_start.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(artifact["status"], "blocked")
            self.assertEqual(artifact["provider"], "cuda")
            self.assertIn(
                "Requested provider `cuda` could not run on the RoboWBC benchmark path.",
                artifact["notes"],
            )
            self.assertIn(
                "provider init failed for cuda: missing runtime dependency libcuda.so.1",
                artifact["notes"],
            )
            self.assertIn("--provider cuda", artifact["source_command"])

    def test_official_gear_sonic_provider_failure_emits_blocked_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            repo_dir = self.make_stub_decoupled_repo(tmp_path)
            gear_sonic_model_dir = self.make_gear_sonic_model_dir(tmp_path)
            fake_harness = self.make_fake_gear_sonic_harness(tmp_path)
            output_dir = tmp_path / "official-artifacts"
            env = os.environ.copy()
            env["GEAR_SONIC_MODEL_DIR"] = str(gear_sonic_model_dir)
            env["GEAR_SONIC_OFFICIAL_HARNESS_BIN"] = str(fake_harness)

            subprocess.run(
                [
                    "python3",
                    str(ROOT / "scripts/bench_nvidia_official.py"),
                    "--case",
                    "gear_sonic/planner_only_cold_start",
                    "--provider",
                    "tensor_rt",
                    "--repo-dir",
                    str(repo_dir),
                    "--output-root",
                    str(output_dir),
                ],
                check=True,
                cwd=ROOT,
                env=env,
            )

            artifact = json.loads(
                (output_dir / "gear_sonic__planner_only_cold_start.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(artifact["status"], "blocked")
            self.assertEqual(artifact["provider"], "tensor_rt")
            self.assertIn(
                "Requested provider `tensor_rt` could not run on the official GEAR-Sonic harness.",
                artifact["notes"],
            )
            self.assertIn(
                "requested provider tensor_rt failed: CUDAExecutionProvider not available",
                artifact["notes"],
            )
            self.assertIn("--provider tensor_rt", artifact["source_command"])

    def test_roboharness_compose_run_config_injects_sections_and_updates_runtime(self) -> None:
        base_toml = textwrap.dedent(
            """
            [policy]
            name = "gear_sonic"

            [runtime]
            max_ticks = 1
            """
        ).strip()
        showcase_context = {
            "transport": "mujoco",
            "model_path": "models/unitree_g1.xml",
            "timestep": 0.002,
            "substeps": 10,
            "gain_profile": "default_pd",
            "robot_config_path": "configs/robots/unitree_g1.toml",
            "config_has_sim_section": False,
        }

        composed = ROBOHARNESS.compose_run_config(
            base_toml,
            Path("/tmp/report.json"),
            Path("/tmp/report_replay_trace.json"),
            Path("/tmp/run.rrd"),
            showcase_context,
            max_ticks=7,
        )

        self.assertIn("[sim]", composed)
        self.assertIn('model_path = "models/unitree_g1.xml"', composed)
        self.assertIn('gain_profile = "default_pd"', composed)
        self.assertIn("[vis]", composed)
        self.assertIn("[report]", composed)
        self.assertIn('output_path = "/tmp/report.json"', composed)
        self.assertIn('save_path = "/tmp/run.rrd"', composed)
        self.assertEqual(composed.count("[runtime]"), 1)
        self.assertIn("max_ticks = 7", composed)

    def test_roboharness_resolve_showcase_context_uses_robot_model_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            robot_dir = repo_root / "configs" / "robots"
            robot_dir.mkdir(parents=True)
            robot_cfg = robot_dir / "unitree_g1_mock.toml"
            robot_cfg.write_text('model_path = "models/unitree_g1.xml"\n', encoding="utf-8")
            config_path = repo_root / "configs" / "sonic_g1.toml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                textwrap.dedent(
                    """
                    [robot]
                    config_path = "configs/robots/unitree_g1_mock.toml"

                    [communication]
                    frequency_hz = 50
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            context = ROBOHARNESS.resolve_showcase_context(repo_root, config_path)

            self.assertEqual(context["transport"], "mujoco")
            self.assertEqual(context["model_path"], "models/unitree_g1.xml")
            self.assertEqual(context["substeps"], 10)
            self.assertEqual(context["gain_profile"], "simulation_pd")
            self.assertFalse(context["config_has_sim_section"])

    def test_roboharness_compose_run_config_preserves_existing_sim_section_and_fills_missing_fields(
        self,
    ) -> None:
        base_toml = textwrap.dedent(
            """
            [policy]
            name = "gear_sonic"

            [sim]
            gain_profile = "default_pd"

            [comm]
            frequency_hz = 50
            """
        ).strip()
        showcase_context = {
            "transport": "mujoco",
            "model_path": "models/unitree_g1.xml",
            "timestep": 0.002,
            "substeps": 10,
            "gain_profile": "simulation_pd",
            "robot_config_path": "configs/robots/unitree_g1.toml",
            "config_has_sim_section": True,
        }

        composed = ROBOHARNESS.compose_run_config(
            base_toml,
            Path("/tmp/report.json"),
            Path("/tmp/report_replay_trace.json"),
            Path("/tmp/run.rrd"),
            showcase_context,
            max_ticks=7,
        )

        parsed = tomllib.loads(composed)
        self.assertEqual(composed.count("[sim]"), 1)
        self.assertIn('model_path = "models/unitree_g1.xml"', composed)
        self.assertIn("timestep = 0.002", composed)
        self.assertIn("substeps = 10", composed)
        self.assertIn('gain_profile = "default_pd"', composed)
        self.assertEqual(parsed["sim"]["gain_profile"], "default_pd")
        self.assertEqual(parsed["sim"]["model_path"], "models/unitree_g1.xml")
        self.assertEqual(parsed["sim"]["timestep"], 0.002)
        self.assertEqual(parsed["sim"]["substeps"], 10)
        self.assertEqual(parsed["comm"]["frequency_hz"], 50)

    def test_roboharness_run_robowbc_records_meta_and_report_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            output_dir = repo_root / "out"
            output_dir.mkdir(parents=True)
            robot_cfg = repo_root / "configs" / "robots" / "unitree_g1_mock.toml"
            robot_cfg.parent.mkdir(parents=True, exist_ok=True)
            robot_cfg.write_text('model_path = "models/unitree_g1.xml"\n', encoding="utf-8")
            config_path = repo_root / "configs" / "sonic_g1.toml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                textwrap.dedent(
                    """
                    [robot]
                    config_path = "configs/robots/unitree_g1_mock.toml"

                    [communication]
                    frequency_hz = 50

                    [sim]
                    model_path = "models/unitree_g1.xml"
                    timestep = 0.002
                    substeps = 10
                    gain_profile = "default_pd"
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            fake_binary = self.make_fake_robowbc_binary(repo_root)

            report = ROBOHARNESS.run_robowbc(
                repo_root=repo_root,
                binary=fake_binary,
                output_dir=output_dir,
                config_path=config_path,
                env=os.environ.copy(),
                max_ticks=3,
            )

            self.assertEqual(report["metrics"]["ticks"], 3)
            self.assertEqual(report["_meta"]["showcase_context"]["transport"], "mujoco")
            self.assertTrue(Path(report["_meta"]["json_path"]).is_file())
            self.assertTrue(Path(report["_meta"]["temp_config"]).is_file())
            temp_config_text = Path(report["_meta"]["temp_config"]).read_text(encoding="utf-8")
            self.assertIn("[report]", temp_config_text)
            self.assertIn("max_ticks = 3", temp_config_text)
            self.assertIn('gain_profile = "default_pd"', temp_config_text)


if __name__ == "__main__":
    unittest.main()
