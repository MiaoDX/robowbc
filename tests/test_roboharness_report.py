import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "scripts" / "reports" / "roboharness_report.py"

SPEC = importlib.util.spec_from_file_location("roboharness_report", REPORT_PATH)
REPORT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REPORT)


def make_frames(count: int, *, control_frequency_hz: int = 50) -> list[dict[str, object]]:
    return [
        {
            "tick": tick,
            "sim_time_secs": tick / control_frequency_hz,
            "target_positions": [0.0, 0.0],
            "actual_positions": [tick * 0.01, -tick * 0.01],
            "actual_velocities": [0.1, -0.1],
            "inference_latency_ms": 1.0 + (tick * 0.05),
            "base_pose": {
                "position_world": [tick * 0.05, 0.0, 0.8],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
        for tick in range(count)
    ]


def make_velocity_timeline() -> list[dict[str, object]]:
    return [
        {
            "phase_name": "stand",
            "start_tick": 0,
            "midpoint_tick": 4,
            "end_tick": 9,
            "duration_ticks": 10,
            "duration_secs": 0.2,
        },
        {
            "phase_name": "run",
            "start_tick": 10,
            "midpoint_tick": 14,
            "end_tick": 19,
            "duration_ticks": 10,
            "duration_secs": 0.2,
        },
    ]


def make_phase_review_checkpoints() -> list[dict[str, object]]:
    return [
        {
            "kind": "phase_midpoint",
            "phase_name": "stand",
            "phase_kind": "midpoint",
            "name": "stand_midpoint",
            "dir": Path("/tmp/stand_midpoint"),
            "relative_dir": "roboharness_run/trial_001/stand_midpoint_tick_0004",
            "meta": {
                "tick": 4,
                "frame_index": 4,
                "sim_time_secs": 0.08,
                "selection_reason": "stand midpoint from the explicit phase timeline",
                "frame_source": "canonical_replay_trace",
                "cameras": ["track", "side", "top"],
            },
        },
        {
            "kind": "phase_end",
            "phase_name": "stand",
            "phase_kind": "phase_end",
            "name": "stand_end",
            "dir": Path("/tmp/stand_end"),
            "relative_dir": "roboharness_run/trial_001/stand_end_tick_0009/lag_3",
            "meta": {
                "tick": 9,
                "frame_index": 9,
                "phase_end_tick": 9,
                "sim_time_secs": 0.18,
                "selection_reason": "stand phase end with positive-lag actual-response review",
                "frame_source": "canonical_replay_trace",
                "cameras": ["track", "side", "top"],
            },
            "lag_options": [0, 1, 2, 3, 4, 5],
            "default_lag_ticks": 3,
            "target_lag_options": [0, 1, 2, 3, 4, 5],
            "default_target_lag_ticks": 0,
            "lag_variants": [
                {
                    "lag_ticks": lag,
                    "lag_ms": lag * 20.0,
                    "tick": 9 + lag,
                    "frame_index": 9 + lag,
                    "sim_time_secs": 0.18 + (lag * 0.02),
                    "selection_reason": f"stand +{lag}",
                    "frame_source": "canonical_replay_trace",
                    "relative_dir": f"roboharness_run/trial_001/stand_end_tick_0009/lag_{lag}",
                    "cameras": ["track", "side", "top"],
                }
                for lag in range(6)
            ],
            "target_lag_variants": [
                {
                    "lag_ticks": lag,
                    "lag_ms": lag * 20.0,
                    "tick": 9 + lag,
                    "frame_index": 9 + lag,
                    "sim_time_secs": 0.18 + (lag * 0.02),
                    "selection_reason": f"stand target +{lag}",
                    "frame_source": "canonical_replay_trace",
                    "relative_dir": f"roboharness_run/trial_001/stand_end_tick_0009/target_lag_{lag}",
                    "cameras": ["track", "side", "top"],
                }
                for lag in range(6)
            ],
        },
        {
            "kind": "diagnostic",
            "name": "peak_latency",
            "dir": Path("/tmp/peak_latency"),
            "relative_dir": "roboharness_run/trial_001/peak_latency_tick_0011",
            "meta": {
                "tick": 11,
                "frame_index": 11,
                "sim_time_secs": 0.22,
                "selection_reason": "peak latency",
                "frame_source": "canonical_replay_trace",
                "cameras": ["track", "side", "top"],
            },
        },
    ]


class RoboharnessReportTests(unittest.TestCase):
    def test_normalize_phase_timeline_entries_derives_midpoint_duration_and_slug(self) -> None:
        normalized = REPORT.normalize_phase_timeline_entries(
            [
                {"phase_name": "Stand", "start_tick": 0, "end_tick": 4},
                {"phase_name": "Accelerate", "start_tick": 5, "end_tick": 14},
            ],
            frame_count=20,
            control_frequency_hz=50,
            source="fixture",
        )

        self.assertEqual(
            normalized,
            [
                {
                    "phase_name": "Stand",
                    "phase_slug": "stand",
                    "start_tick": 0,
                    "midpoint_tick": 2,
                    "end_tick": 4,
                    "duration_ticks": 5,
                    "duration_secs": 0.1,
                },
                {
                    "phase_name": "Accelerate",
                    "phase_slug": "accelerate",
                    "start_tick": 5,
                    "midpoint_tick": 9,
                    "end_tick": 14,
                    "duration_ticks": 10,
                    "duration_secs": 0.2,
                },
            ],
        )

    def test_normalize_phase_timeline_entries_rejects_hostile_phase_names(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            REPORT.normalize_phase_timeline_entries(
                [{"phase_name": "../escape", "start_tick": 0, "end_tick": 0}],
                frame_count=1,
                control_frequency_hz=50,
                source="fixture",
            )

        self.assertIn("must not contain path separators", str(ctx.exception))

    def test_resolve_phase_review_contract_uses_velocity_schedule_artifact(self) -> None:
        report = {
            "command_kind": "velocity_schedule",
            "control_frequency_hz": 50,
            "frames": make_frames(26),
            "phase_timeline": make_velocity_timeline(),
        }

        contract = REPORT.resolve_phase_review_contract(ROOT, report)

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(contract["source"], "velocity_schedule")
        self.assertEqual(contract["default_lag_ticks"], 3)
        self.assertEqual(contract["default_lag_ms"], 60.0)
        self.assertEqual(contract["lag_options"], [0, 1, 2, 3, 4, 5])
        self.assertEqual(contract["default_target_lag_ticks"], 0)
        self.assertEqual(contract["default_target_lag_ms"], 0.0)
        self.assertEqual(contract["target_lag_options"], [0, 1, 2, 3, 4, 5])
        self.assertEqual(
            [entry["phase_name"] for entry in contract["phase_timeline"]],
            ["stand", "run"],
        )

    def test_resolve_phase_review_contract_returns_none_for_tracking_without_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            config_path = repo_root / "configs" / "tracking.toml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("[runtime]\n", encoding="utf-8")

            contract = REPORT.resolve_phase_review_contract(
                repo_root,
                {
                    "command_kind": "reference_motion_tracking",
                    "control_frequency_hz": 50,
                    "frames": make_frames(20),
                    "_meta": {"config_path": "configs/tracking.toml"},
                },
            )

        self.assertIsNone(contract)

    def test_resolve_phase_review_contract_loads_tracking_sidecar_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            config_path = repo_root / "configs" / "tracking.toml"
            sidecar_path = repo_root / "configs" / "tracking.phases.toml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("[runtime]\n", encoding="utf-8")
            sidecar_path.write_text(
                "\n".join(
                    [
                        "default_lag_ticks = 2",
                        "",
                        "[[phases]]",
                        'phase_name = "lift"',
                        "start_tick = 0",
                        "end_tick = 4",
                        "",
                        "[[phases]]",
                        'phase_name = "place"',
                        "start_tick = 5",
                        "end_tick = 9",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            contract = REPORT.resolve_phase_review_contract(
                repo_root,
                {
                    "command_kind": "reference_motion_tracking",
                    "control_frequency_hz": 100,
                    "frames": make_frames(20, control_frequency_hz=100),
                    "_meta": {"config_path": "configs/tracking.toml"},
                },
            )

        self.assertIsNotNone(contract)
        assert contract is not None
        self.assertEqual(contract["source"], "tracking_sidecar")
        self.assertEqual(contract["default_lag_ticks"], 2)
        self.assertEqual(contract["default_lag_ms"], 20.0)
        self.assertEqual(contract["default_target_lag_ticks"], 0)
        self.assertEqual(contract["default_target_lag_ms"], 0.0)
        self.assertEqual(
            [entry["phase_name"] for entry in contract["phase_timeline"]],
            ["lift", "place"],
        )

    def test_load_tracking_phase_review_contract_rejects_invalid_sidecars(self) -> None:
        cases = {
            "default_lag": (
                "\n".join(
                    [
                        "default_lag_ticks = 6",
                        "",
                        "[[phases]]",
                        'phase_name = "lift"',
                        "start_tick = 0",
                        "end_tick = 4",
                        "",
                    ]
                ),
                "must be an integer between 0 and 5",
            ),
            "duplicate_name": (
                "\n".join(
                    [
                        "[[phases]]",
                        'phase_name = "lift"',
                        "start_tick = 0",
                        "end_tick = 4",
                        "",
                        "[[phases]]",
                        'phase_name = "lift"',
                        "start_tick = 5",
                        "end_tick = 9",
                        "",
                    ]
                ),
                "duplicate phase name",
            ),
            "overlap": (
                "\n".join(
                    [
                        "[[phases]]",
                        'phase_name = "lift"',
                        "start_tick = 0",
                        "end_tick = 4",
                        "",
                        "[[phases]]",
                        'phase_name = "place"',
                        "start_tick = 4",
                        "end_tick = 9",
                        "",
                    ]
                ),
                "overlaps or is out of order",
            ),
            "out_of_bounds": (
                "\n".join(
                    [
                        "[[phases]]",
                        'phase_name = "lift"',
                        "start_tick = 0",
                        "end_tick = 99",
                        "",
                    ]
                ),
                "only 20 frames were recorded",
            ),
            "path_escape_name": (
                "\n".join(
                    [
                        "[[phases]]",
                        'phase_name = "../escape"',
                        "start_tick = 0",
                        "end_tick = 4",
                        "",
                    ]
                ),
                "must not contain path separators",
            ),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            for label, (text, expected) in cases.items():
                with self.subTest(case=label):
                    sidecar_path = tmp_root / f"{label}.phases.toml"
                    sidecar_path.write_text(text, encoding="utf-8")
                    with self.assertRaises(SystemExit) as ctx:
                        REPORT.load_tracking_phase_review_contract(
                            sidecar_path,
                            frame_count=20,
                            control_frequency_hz=50,
                        )
                    self.assertIn(expected, str(ctx.exception))

    def test_resolve_phase_review_contract_rejects_config_paths_outside_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            with self.assertRaises(SystemExit) as ctx:
                REPORT.resolve_phase_review_contract(
                    repo_root,
                    {
                        "command_kind": "reference_motion_tracking",
                        "frames": make_frames(10),
                        "_meta": {"config_path": "../outside.toml"},
                    },
                )

        self.assertIn("must stay inside the repo root", str(ctx.exception))

    def test_build_phase_review_capture_plan_bounds_positive_lag_fanout(self) -> None:
        plan = REPORT.build_phase_review_capture_plan(
            make_frames(12),
            {
                "phase_timeline": [
                    {
                        "phase_name": "settle",
                        "phase_slug": "settle",
                        "start_tick": 0,
                        "midpoint_tick": 4,
                        "end_tick": 9,
                        "duration_ticks": 10,
                        "duration_secs": 0.2,
                    }
                ],
                "lag_options": [0, 1, 2, 3, 4, 5, 8, -1],
                "default_lag_ticks": 3,
                "target_lag_options": [0, 1, 2, 3, 4, 5, 8, -1],
                "default_target_lag_ticks": 0,
            },
            frame_source="canonical_replay_trace",
            control_frequency_hz=50,
        )

        self.assertEqual(len(plan), 2)
        midpoint, phase_end = plan
        self.assertEqual(midpoint["kind"], "phase_midpoint")
        self.assertEqual(phase_end["kind"], "phase_end")
        self.assertEqual(phase_end["lag_options"], [0, 1, 2])
        self.assertEqual(phase_end["default_display_lag"], 2)
        self.assertEqual(phase_end["target_lag_options"], [0, 1, 2])
        self.assertEqual(phase_end["default_target_display_lag"], 0)
        self.assertEqual(
            [variant["lag_ticks"] for variant in phase_end["variants"]],
            [0, 1, 2],
        )
        self.assertEqual(
            [variant["tick"] for variant in phase_end["variants"]],
            [9, 10, 11],
        )

    def test_build_phase_review_capture_plan_scales_linearly_for_many_phases(self) -> None:
        timeline = [
            {
                "phase_name": f"phase_{index}",
                "phase_slug": f"phase-{index}",
                "start_tick": index * 2,
                "midpoint_tick": index * 2,
                "end_tick": (index * 2) + 1,
                "duration_ticks": 2,
                "duration_secs": 0.04,
            }
            for index in range(10)
        ]

        plan = REPORT.build_phase_review_capture_plan(
            make_frames(26),
            {
                "phase_timeline": timeline,
                "lag_options": [0, 1, 2, 3, 4, 5],
                "default_lag_ticks": 3,
                "target_lag_options": [0, 1, 2, 3, 4, 5],
                "default_target_lag_ticks": 0,
            },
            frame_source="canonical_replay_trace",
            control_frequency_hz=50,
        )

        phase_end_entries = [entry for entry in plan if entry["kind"] == "phase_end"]
        self.assertEqual(len(plan), 20)
        self.assertEqual(len(phase_end_entries), 10)
        self.assertTrue(all(len(entry["variants"]) <= 6 for entry in phase_end_entries))
        self.assertEqual(
            sum(len(entry["variants"]) for entry in phase_end_entries),
            60,
        )

    def test_build_proof_pack_manifest_payload_emits_phase_review_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            report = {
                "policy_name": "gear_sonic",
                "robot_name": "unitree_g1",
                "command_kind": "velocity_schedule",
                "command_data": [0.6, 0.0, 0.0],
                "control_frequency_hz": 50,
                "transport": "mujoco",
                "frames": make_frames(26),
                "_meta": {
                    "json_path": str(output_dir / "run_report.json"),
                    "replay_trace_path": str(output_dir / "run_report_replay_trace.json"),
                    "rrd_path": str(output_dir / "run_recording.rrd"),
                    "log_path": str(output_dir / "run.log"),
                    "temp_config": str(output_dir / "roboharness_run.toml"),
                },
                "_phase_review": {
                    "source": "velocity_schedule",
                    "default_lag_ticks": 3,
                    "default_lag_ms": 60.0,
                    "lag_options": [0, 1, 2, 3, 4, 5],
                    "default_target_lag_ticks": 0,
                    "default_target_lag_ms": 0.0,
                    "target_lag_options": [0, 1, 2, 3, 4, 5],
                    "phase_timeline": REPORT.normalize_phase_timeline_entries(
                        make_velocity_timeline(),
                        frame_count=26,
                        control_frequency_hz=50,
                        source="fixture",
                    ),
                },
            }

            payload = REPORT.build_proof_pack_manifest_payload(
                output_dir,
                report,
                make_phase_review_checkpoints(),
                html_entrypoint="index.html",
            )

        self.assertEqual(payload["capture_status"], "ok")
        self.assertEqual(payload["phase_review"]["version"], 1)
        self.assertEqual(payload["phase_review"]["source"], "velocity_schedule")
        self.assertEqual(payload["lag_options"], [0, 1, 2, 3, 4, 5])
        self.assertEqual(payload["default_lag_ticks"], 3)
        self.assertEqual(payload["default_lag_ms"], 60.0)
        self.assertEqual(payload["target_lag_options"], [0, 1, 2, 3, 4, 5])
        self.assertEqual(payload["default_target_lag_ticks"], 0)
        self.assertEqual(payload["default_target_lag_ms"], 0.0)
        self.assertEqual(len(payload["phase_timeline"]), 2)
        self.assertEqual(len(payload["phase_checkpoints"]), 2)
        self.assertEqual(len(payload["diagnostic_checkpoints"]), 1)
        self.assertEqual(len(payload["checkpoints"]), 3)
        self.assertEqual(payload["raw_artifacts"]["run_report"], "run_report.json")
        self.assertEqual(payload["raw_artifacts"]["replay_trace"], "run_report_replay_trace.json")

    def test_build_proof_pack_manifest_payload_keeps_generic_contract_without_phase_review(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            payload = REPORT.build_proof_pack_manifest_payload(
                output_dir,
                {
                    "policy_name": "bfm_zero",
                    "robot_name": "unitree_g1",
                    "command_kind": "motion_tokens",
                    "command_data": [0.0],
                    "control_frequency_hz": 50,
                    "transport": "mujoco",
                    "frames": make_frames(5),
                    "_meta": {},
                },
                [
                    {
                        "kind": "diagnostic",
                        "name": "start",
                        "dir": Path("/tmp/start"),
                        "relative_dir": "roboharness_run/trial_001/start_tick_0000",
                        "meta": {
                            "tick": 0,
                            "frame_index": 0,
                            "sim_time_secs": 0.0,
                            "selection_reason": "initial state",
                            "frame_source": "run_report_frames",
                            "cameras": ["track", "side", "top"],
                        },
                    }
                ],
                html_entrypoint="index.html",
            )

        self.assertEqual(payload["capture_status"], "ok")
        self.assertIn("checkpoints", payload)
        self.assertNotIn("phase_review", payload)
        self.assertNotIn("phase_checkpoints", payload)
        self.assertNotIn("diagnostic_checkpoints", payload)


if __name__ == "__main__":
    unittest.main()
