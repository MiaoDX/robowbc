import importlib.util
import json
import tempfile
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SHOWCASE_PATH = ROOT / "scripts" / "site" / "generate_policy_showcase.py"

SHOWCASE_SPEC = importlib.util.spec_from_file_location(
    "generate_policy_showcase", SHOWCASE_PATH
)
SHOWCASE = importlib.util.module_from_spec(SHOWCASE_SPEC)
assert SHOWCASE_SPEC.loader is not None
SHOWCASE_SPEC.loader.exec_module(SHOWCASE)


def make_frames() -> list[dict[str, object]]:
    return [
        {
            "tick": 0,
            "target_positions": [0.0],
            "actual_positions": [0.0],
            "actual_velocities": [0.0],
            "command_data": [0.2, 0.0, 0.0],
            "inference_latency_ms": 1.0,
            "base_pose": {
                "position_world": [0.0, 0.0, 0.8],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
        {
            "tick": 1,
            "target_positions": [0.1],
            "actual_positions": [0.1],
            "actual_velocities": [0.1],
            "command_data": [0.2, 0.0, 0.0],
            "inference_latency_ms": 1.1,
            "base_pose": {
                "position_world": [0.01, 0.0, 0.8],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
        {
            "tick": 2,
            "target_positions": [0.2],
            "actual_positions": [0.2],
            "actual_velocities": [0.1],
            "command_data": [0.2, 0.0, 0.0],
            "inference_latency_ms": 1.2,
            "base_pose": {
                "position_world": [0.02, 0.0, 0.8],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
    ]


def make_bad_tracking_frames() -> list[dict[str, object]]:
    return [
        {
            "tick": 0,
            "target_positions": [0.0, 0.0],
            "actual_positions": [1.8, -1.6],
            "actual_velocities": [0.0, 0.0],
            "command_data": [],
            "inference_latency_ms": 3.0,
            "base_pose": {
                "position_world": [0.0, 0.0, 0.75],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
        {
            "tick": 1,
            "target_positions": [0.1, -0.1],
            "actual_positions": [2.0, -1.7],
            "actual_velocities": [0.1, -0.1],
            "command_data": [],
            "inference_latency_ms": 3.2,
            "base_pose": {
                "position_world": [0.0, 0.0, 0.22],
                "rotation_xyzw": [0.0, 0.0, 0.2, 0.98],
            },
        },
        {
            "tick": 2,
            "target_positions": [0.2, -0.2],
            "actual_positions": [2.1, -1.8],
            "actual_velocities": [0.1, -0.1],
            "command_data": [],
            "inference_latency_ms": 3.4,
            "base_pose": {
                "position_world": [0.0, 0.0, 0.18],
                "rotation_xyzw": [0.0, 0.0, 0.4, 0.92],
            },
        },
    ]


def make_entry(
    *,
    card_id: str,
    policy_family: str,
    title: str,
    command_kind: str,
    command_source: str,
    demo_family: str,
    rrd_file: str,
    velocity_tracking: dict[str, float] | None = None,
    frames: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "ticks": 3,
        "average_inference_ms": 1.25,
        "achieved_frequency_hz": 48.5,
        "dropped_frames": 0,
    }
    if velocity_tracking is not None:
        metrics["velocity_tracking"] = velocity_tracking

    return {
        "card_id": card_id,
        "policy_name": policy_family,
        "robot_name": "unitree_g1",
        "command_kind": command_kind,
        "command_data": [0.2, 0.0, 0.0],
        "control_frequency_hz": 50,
        "joint_names": ["left_hip_pitch_joint"],
        "frames": frames if frames is not None else make_frames(),
        "metrics": metrics,
        "_meta": {
            "card_id": card_id,
            "policy_family": policy_family,
            "title": title,
            "source": "NVIDIA GR00T",
            "summary": "Synthetic test entry for showcase rendering.",
            "coverage": "Expected behavior for the selected demo contract.",
            "execution_kind": "real",
            "checkpoint_source": "Test fixture",
            "command_source": command_source,
            "demo_family": demo_family,
            "demo_sequence": "Synthetic scenario for HTML regression coverage.",
            "model_artifact": "models/test.onnx",
            "config_path": "configs/test.toml",
            "required_paths": [],
            "blocked_reason": None,
            "showcase_transport": "mujoco",
            "showcase_model_path": "models/unitree_g1.xml",
            "showcase_gain_profile": "simulation_pd",
            "showcase_model_variant": "meshless-public-mjcf",
            "robot_config_path": "configs/robots/unitree_g1.toml",
            "json_file": f"{card_id}.json",
            "rrd_file": rrd_file,
            "log_file": f"{card_id}.log",
        },
    }


def make_phase_review_manifest() -> dict[str, object]:
    return {
        "capture_status": "ok",
        "phase_review": {
            "enabled": True,
            "version": 1,
            "source": "velocity_schedule",
        },
        "phase_timeline": [
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
        ],
        "phase_checkpoints": [
            {
                "name": "stand_midpoint",
                "phase_name": "stand",
                "phase_kind": "midpoint",
                "relative_dir": "roboharness_run/trial_001/stand_midpoint_tick_0004",
                "tick": 4,
                "frame_index": 4,
                "sim_time_secs": 0.08,
                "selection_reason": "stand midpoint",
                "frame_source": "canonical_replay_trace",
                "cameras": ["track", "side", "top"],
            },
            {
                "name": "stand_end",
                "phase_name": "stand",
                "phase_kind": "phase_end",
                "relative_dir": "roboharness_run/trial_001/stand_end_tick_0009/lag_3",
                "tick": 9,
                "frame_index": 9,
                "phase_end_tick": 9,
                "sim_time_secs": 0.18,
                "selection_reason": "stand phase end",
                "frame_source": "canonical_replay_trace",
                "cameras": ["track", "side", "top"],
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
                "name": "run_midpoint",
                "phase_name": "run",
                "phase_kind": "midpoint",
                "relative_dir": "roboharness_run/trial_001/run_midpoint_tick_0014",
                "tick": 14,
                "frame_index": 14,
                "sim_time_secs": 0.28,
                "selection_reason": "run midpoint",
                "frame_source": "canonical_replay_trace",
                "cameras": ["track", "side", "top"],
            },
            {
                "name": "run_end",
                "phase_name": "run",
                "phase_kind": "phase_end",
                "relative_dir": "roboharness_run/trial_001/run_end_tick_0019/lag_3",
                "tick": 19,
                "frame_index": 19,
                "phase_end_tick": 19,
                "sim_time_secs": 0.38,
                "selection_reason": "run phase end",
                "frame_source": "canonical_replay_trace",
                "cameras": ["track", "side", "top"],
                "lag_options": [0, 1, 2, 3, 4, 5],
                "default_lag_ticks": 3,
                "target_lag_options": [0, 1, 2, 3, 4, 5],
                "default_target_lag_ticks": 0,
                "lag_variants": [
                    {
                        "lag_ticks": lag,
                        "lag_ms": lag * 20.0,
                        "tick": 19 + lag,
                        "frame_index": 19 + lag,
                        "sim_time_secs": 0.38 + (lag * 0.02),
                        "selection_reason": f"run +{lag}",
                        "frame_source": "canonical_replay_trace",
                        "relative_dir": f"roboharness_run/trial_001/run_end_tick_0019/lag_{lag}",
                        "cameras": ["track", "side", "top"],
                    }
                    for lag in range(6)
                ],
                "target_lag_variants": [
                    {
                        "lag_ticks": lag,
                        "lag_ms": lag * 20.0,
                        "tick": 19 + lag,
                        "frame_index": 19 + lag,
                        "sim_time_secs": 0.38 + (lag * 0.02),
                        "selection_reason": f"run target +{lag}",
                        "frame_source": "canonical_replay_trace",
                        "relative_dir": f"roboharness_run/trial_001/run_end_tick_0019/target_lag_{lag}",
                        "cameras": ["track", "side", "top"],
                    }
                    for lag in range(6)
                ],
            },
        ],
        "diagnostic_checkpoints": [
            {
                "name": "peak_latency",
                "relative_dir": "roboharness_run/trial_001/peak_latency_tick_0011",
                "tick": 11,
                "frame_index": 11,
                "sim_time_secs": 0.22,
                "selection_reason": "peak latency",
                "frame_source": "canonical_replay_trace",
                "cameras": ["track", "side", "top"],
            }
        ],
        "lag_options": [0, 1, 2, 3, 4, 5],
        "default_lag_ticks": 3,
        "default_lag_ms": 60.0,
        "target_lag_options": [0, 1, 2, 3, 4, 5],
        "default_target_lag_ticks": 0,
        "default_target_lag_ms": 0.0,
    }


class PolicyShowcaseTests(unittest.TestCase):
    def render(
        self, entries: list[dict[str, object]]
    ) -> tuple[str, list[dict[str, object]], dict[str, str]]:
        original_vendor = SHOWCASE.vendor_rerun_web_viewer
        SHOWCASE.vendor_rerun_web_viewer = lambda repo_root, output_dir: {
            "version": "test",
            "module_path": "./_rerun_web_viewer/index.js",
        }
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                SHOWCASE.render_html(entries, output_dir, ROOT)
                html_text = (output_dir / "index.html").read_text(encoding="utf-8")
                manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
                detail_pages = {
                    str(entry["card_id"]): (
                        output_dir / "policies" / str(entry["card_id"]) / "index.html"
                    ).read_text(encoding="utf-8")
                    for entry in manifest
                }
                return html_text, manifest, detail_pages
        finally:
            SHOWCASE.vendor_rerun_web_viewer = original_vendor

    def test_render_html_uses_unique_card_ids_and_viewer_keys_for_duplicate_policy_families(
        self,
    ) -> None:
        entries = [
            make_entry(
                card_id="gear_sonic",
                policy_family="gear_sonic",
                title="GEAR-SONIC",
                command_kind="velocity_schedule",
                command_source="runtime.velocity_schedule",
                demo_family="Velocity tracking",
                rrd_file="gear_sonic.rrd",
                velocity_tracking={
                    "vx_rmse_mps": 0.2,
                    "yaw_rate_rmse_rad_s": 0.3,
                    "heading_change_deg": -90.0,
                    "forward_distance_m": 3.0,
                },
            ),
            make_entry(
                card_id="gear_sonic_tracking",
                policy_family="gear_sonic",
                title="GEAR-SONIC Reference Motion",
                command_kind="reference_motion_tracking",
                command_source="runtime.reference_motion_tracking",
                demo_family="Reference / pose tracking",
                rrd_file="gear_sonic_tracking.rrd",
            ),
        ]

        html_text, manifest, detail_pages = self.render(entries)

        self.assertIn('href="policies/gear_sonic/"', html_text)
        self.assertIn('href="policies/gear_sonic_tracking/"', html_text)
        self.assertIn('id="policy-gear_sonic"', detail_pages["gear_sonic"])
        self.assertIn('id="policy-gear_sonic_tracking"', detail_pages["gear_sonic_tracking"])
        self.assertIn('data-rerun-policy="gear_sonic"', detail_pages["gear_sonic"])
        self.assertIn(
            'data-rerun-policy="gear_sonic_tracking"',
            detail_pages["gear_sonic_tracking"],
        )

        self.assertIn("gear_sonic_tracking · family gear_sonic", html_text)
        self.assertEqual([entry["card_id"] for entry in manifest], ["gear_sonic", "gear_sonic_tracking"])

    def test_render_html_recovers_unique_card_ids_from_artifact_filenames(self) -> None:
        gear_sonic = make_entry(
            card_id="gear_sonic",
            policy_family="gear_sonic",
            title="GEAR-SONIC",
            command_kind="velocity_schedule",
            command_source="runtime.velocity_schedule",
            demo_family="Velocity tracking",
            rrd_file="gear_sonic.rrd",
        )
        gear_sonic_tracking = make_entry(
            card_id="gear_sonic_tracking",
            policy_family="gear_sonic",
            title="GEAR-SONIC Reference Motion",
            command_kind="reference_motion_tracking",
            command_source="runtime.reference_motion_tracking",
            demo_family="Reference / pose tracking",
            rrd_file="gear_sonic_tracking.rrd",
        )
        for entry in (gear_sonic, gear_sonic_tracking):
            entry.pop("card_id")
            entry["_meta"].pop("card_id")

        html_text, manifest, detail_pages = self.render([gear_sonic, gear_sonic_tracking])

        self.assertIn('href="policies/gear_sonic/"', html_text)
        self.assertIn('href="policies/gear_sonic_tracking/"', html_text)
        self.assertEqual([entry["card_id"] for entry in manifest], ["gear_sonic", "gear_sonic_tracking"])
        self.assertIn('data-rerun-policy="gear_sonic"', detail_pages["gear_sonic"])
        self.assertIn(
            'data-rerun-policy="gear_sonic_tracking"',
            detail_pages["gear_sonic_tracking"],
        )

    def test_render_html_shows_velocity_metrics_only_for_velocity_cards(self) -> None:
        entries = [
            make_entry(
                card_id="decoupled_wbc",
                policy_family="decoupled_wbc",
                title="Decoupled WBC",
                command_kind="velocity_schedule",
                command_source="runtime.velocity_schedule",
                demo_family="Velocity tracking",
                rrd_file="decoupled_wbc.rrd",
                velocity_tracking={
                    "vx_rmse_mps": 0.165,
                    "yaw_rate_rmse_rad_s": 0.304,
                    "heading_change_deg": -83.8,
                    "forward_distance_m": 3.89,
                },
            ),
            make_entry(
                card_id="bfm_zero",
                policy_family="bfm_zero",
                title="BFM-Zero",
                command_kind="motion_tokens",
                command_source="runtime.motion_tokens",
                demo_family="Reference / pose tracking",
                rrd_file="bfm_zero.rrd",
            ),
        ]

        html_text, _manifest, detail_pages = self.render(entries)
        velocity_detail = detail_pages["decoupled_wbc"]
        tracking_detail = detail_pages["bfm_zero"]

        self.assertIn("VX RMSE", velocity_detail)
        self.assertIn("Yaw RMSE", velocity_detail)
        self.assertNotIn("VX RMSE", tracking_detail)
        self.assertNotIn("Yaw RMSE", tracking_detail)
        self.assertIn("Mean joint error", velocity_detail)
        self.assertIn("Mean joint error", tracking_detail)
        self.assertIn("Quality verdict", velocity_detail)
        self.assertIn("Quality verdict", tracking_detail)
        self.assertIn(">MOTION_TOKENS<", html_text)
        self.assertIn(">VELOCITY_SCHEDULE<", html_text)

    def test_render_html_marks_bad_tracking_cards_with_quality_pill_and_heuristics(self) -> None:
        entries = [
            make_entry(
                card_id="gear_sonic_tracking",
                policy_family="gear_sonic",
                title="GEAR-SONIC Reference Motion",
                command_kind="reference_motion_tracking",
                command_source="runtime.reference_motion_tracking",
                demo_family="Reference / pose tracking",
                rrd_file="gear_sonic_tracking.rrd",
                frames=make_bad_tracking_frames(),
            )
        ]

        html_text, manifest, detail_pages = self.render(entries)
        tracking_detail = detail_pages["gear_sonic_tracking"]

        self.assertIn(">BAD<", html_text)
        self.assertIn("Mean joint error", tracking_detail)
        self.assertIn("Frames height &lt; 0.4 m", tracking_detail)
        self.assertIn("Min base height", tracking_detail)
        self.assertEqual(manifest[0]["quality_verdict"]["label"], "BAD")
        self.assertIn("mean joint error", manifest[0]["quality_verdict"]["summary"])

    def test_compose_showcase_config_preserves_existing_sim_section_and_fills_missing_fields(
        self,
    ) -> None:
        base_toml = """
[policy]
name = "gear_sonic"

[sim]
gain_profile = "default_pd"

[comm]
frequency_hz = 50
""".strip()
        showcase_context = {
            "transport": "mujoco",
            "model_path": "models/unitree_g1.xml",
            "timestep": 0.002,
            "substeps": 10,
            "gain_profile": "simulation_pd",
            "report_max_frames": 200,
        }

        composed = SHOWCASE.compose_showcase_config(
            base_toml,
            "gear_sonic",
            Path("/tmp/report.json"),
            Path("/tmp/run.rrd"),
            showcase_context,
        )

        parsed = tomllib.loads(composed)
        self.assertEqual(composed.count("[sim]"), 1)
        self.assertEqual(parsed["sim"]["gain_profile"], "default_pd")
        self.assertEqual(parsed["sim"]["model_path"], "models/unitree_g1.xml")
        self.assertEqual(parsed["sim"]["timestep"], 0.002)
        self.assertEqual(parsed["sim"]["substeps"], 10)
        self.assertEqual(parsed["comm"]["frequency_hz"], 50)
        self.assertEqual(parsed["vis"]["app_id"], "robowbc-showcase-gear_sonic")
        self.assertEqual(parsed["vis"]["save_path"], "/tmp/run.rrd")
        self.assertEqual(parsed["report"]["output_path"], "/tmp/report.json")

    def test_classify_quality_verdict_marks_straight_velocity_run_as_good(self) -> None:
        verdict = SHOWCASE.classify_quality_verdict(
            "ok",
            "velocity_schedule",
            {
                "dropped_frames": 1,
                "achieved_frequency_hz": 47.5,
                "velocity_tracking": {
                    "vx_rmse_mps": 0.346,
                    "yaw_rate_rmse_rad_s": 0.894,
                    "heading_change_deg": 2.4,
                    "forward_distance_m": 2.91,
                },
                "target_tracking": {
                    "frames_below_base_height_0_4m": 0,
                    "frames_below_base_height_0_2m": 0,
                },
            },
        )

        self.assertEqual(verdict["label"], "GOOD")
        self.assertEqual(verdict["css_class"], "good")

    def test_classify_quality_verdict_marks_stable_tracking_as_good(self) -> None:
        verdict = SHOWCASE.classify_quality_verdict(
            "ok",
            "motion_tokens",
            {
                "dropped_frames": 0,
                "achieved_frequency_hz": 47.6,
                "target_tracking": {
                    "mean_joint_abs_error_rad": 0.13,
                    "p95_joint_abs_error_rad": 0.42,
                    "base_height_min_m": 0.74,
                    "frames_below_base_height_0_4m": 0,
                    "frames_below_base_height_0_2m": 0,
                },
            },
        )

        self.assertEqual(verdict["label"], "GOOD")
        self.assertEqual(verdict["css_class"], "good")

    def test_render_html_marks_mixed_velocity_cards_as_unknown(self) -> None:
        entry = make_entry(
            card_id="wbc_agile",
            policy_family="wbc_agile",
            title="WBC-AGILE",
            command_kind="velocity_schedule",
            command_source="runtime.velocity_schedule",
            demo_family="Velocity tracking",
            rrd_file="wbc_agile.rrd",
            velocity_tracking={
                "vx_rmse_mps": 0.39,
                "yaw_rate_rmse_rad_s": 0.37,
                "heading_change_deg": -96.0,
                "forward_distance_m": 1.86,
            },
        )

        html_text, manifest, _detail_pages = self.render([entry])

        self.assertIn(">??<", html_text)
        self.assertEqual(manifest[0]["quality_verdict"]["label"], "??")
        self.assertIn("forward distance", manifest[0]["quality_verdict"]["summary"])

    def test_render_html_renders_phase_review_contract_for_phase_aware_proof_pack(self) -> None:
        entry = make_entry(
            card_id="gear_sonic",
            policy_family="gear_sonic",
            title="GEAR-SONIC",
            command_kind="velocity_schedule",
            command_source="runtime.velocity_schedule",
            demo_family="Velocity tracking",
            rrd_file="gear_sonic.rrd",
        )
        entry["_meta"]["proof_pack_manifest_file"] = "policies/gear_sonic/proof_pack_manifest.json"
        entry["_proof_pack_manifest"] = make_phase_review_manifest()

        _html_text, _manifest, detail_pages = self.render([entry])
        detail = detail_pages["gear_sonic"]

        self.assertIn('id="phase-timeline"', detail)
        self.assertIn('id="phase-lag-selector"', detail)
        self.assertIn('id="phase-target-lag-selector"', detail)
        self.assertIn('data-default-lag="3"', detail)
        self.assertIn('data-default-lag="0"', detail)
        self.assertIn('data-selected-lag="3"', detail)
        self.assertIn('data-selected-lag="0"', detail)
        self.assertIn('data-active="true"', detail)
        self.assertIn('aria-pressed="true"', detail)
        for lag in range(6):
            self.assertIn(f'data-lag="{lag}"', detail)
            self.assertIn(f'>{f"+{lag}"}<', detail)
        self.assertIn("T+0 (0 ms)", detail)
        self.assertIn("A+3 (60 ms)", detail)
        self.assertIn("stand", detail)
        self.assertIn("run", detail)
        self.assertIn('data-phase-debug-phase="stand"', detail)
        self.assertIn("actual_variants", detail)
        self.assertIn("target_variants", detail)
        self.assertIn("Diagnostics", detail)
        self.assertIn("peak_latency", detail)


if __name__ == "__main__":
    unittest.main()
