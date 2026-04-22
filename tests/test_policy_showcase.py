import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHOWCASE_PATH = ROOT / "scripts" / "generate_policy_showcase.py"

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
        "frames": make_frames(),
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


class PolicyShowcaseTests(unittest.TestCase):
    def render(self, entries: list[dict[str, object]]) -> tuple[str, list[dict[str, object]]]:
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
                return html_text, manifest
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

        html_text, manifest = self.render(entries)

        section_ids = re.findall(r'<section class="card" id="policy-([^"]+)"', html_text)
        self.assertEqual(section_ids, ["gear_sonic", "gear_sonic_tracking"])

        viewer_keys = re.findall(r'data-rerun-policy="([^"]+)"', html_text)
        self.assertEqual(viewer_keys, ["gear_sonic", "gear_sonic_tracking"])

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

        html_text, _manifest = self.render([gear_sonic, gear_sonic_tracking])

        section_ids = re.findall(r'<section class="card" id="policy-([^"]+)"', html_text)
        viewer_keys = re.findall(r'data-rerun-policy="([^"]+)"', html_text)

        self.assertEqual(section_ids, ["gear_sonic", "gear_sonic_tracking"])
        self.assertEqual(viewer_keys, ["gear_sonic", "gear_sonic_tracking"])

    def test_render_html_only_shows_velocity_tracking_metrics_for_velocity_cards(self) -> None:
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

        html_text, _manifest = self.render(entries)

        self.assertEqual(html_text.count("VX RMSE"), 1)
        self.assertEqual(html_text.count("Yaw RMSE"), 1)
        self.assertIn(">motion_tokens<", html_text)
        self.assertIn(">velocity_schedule<", html_text)


if __name__ == "__main__":
    unittest.main()
