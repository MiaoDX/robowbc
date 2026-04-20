import importlib.util
import json
import subprocess
import tempfile
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

    def test_official_wrapper_emits_blocked_artifacts_for_every_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [
                    "bash",
                    str(ROOT / "scripts/bench_nvidia_official.sh"),
                    "--all",
                    "--output-root",
                    tmpdir,
                ],
                check=True,
                cwd=ROOT,
            )
            registry = NORMALIZER.load_registry(REGISTRY_PATH)
            for case in registry["cases"]:
                artifact_path = Path(tmpdir) / f"{case['case_id'].replace('/', '__')}.json"
                self.assertTrue(artifact_path.is_file(), msg=f"missing {artifact_path}")
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                self.assertEqual(artifact["case_id"], case["case_id"])
                self.assertEqual(artifact["status"], "blocked")
                self.assertEqual(artifact["stack"], "official_nvidia")


if __name__ == "__main__":
    unittest.main()
