import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = ROOT / "scripts" / "site" / "validate_site_bundle.py"

SPEC = importlib.util.spec_from_file_location("validate_site_bundle", VALIDATOR_PATH)
VALIDATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VALIDATOR)


def write_rgb_images(root: Path, relative_dir: str, cameras: list[str]) -> None:
    checkpoint_dir = root / relative_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        (checkpoint_dir / f"{camera}_rgb.png").write_bytes(b"png")


def write_phase_end_variant_images(root: Path, relative_dir: str, cameras: list[str]) -> None:
    checkpoint_dir = root / relative_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        (checkpoint_dir / f"{camera}_rgb.png").write_bytes(b"png")
        (checkpoint_dir / f"{camera}_actual_rgb.png").write_bytes(b"png")
        (checkpoint_dir / f"{camera}_target_rgb.png").write_bytes(b"png")


def make_generic_proof_pack_manifest(*, capture_status: str = "ok") -> dict[str, object]:
    checkpoints: list[dict[str, object]]
    if capture_status == "ok":
        checkpoints = [
            {
                "name": "start_tick_0000",
                "relative_dir": "roboharness_run/trial_001/start_tick_0000",
                "cameras": ["track", "side", "top"],
            }
        ]
    else:
        checkpoints = []

    return {
        "capture_status": capture_status,
        "checkpoints": checkpoints,
    }


def make_phase_review_manifest() -> dict[str, object]:
    return {
        "capture_status": "ok",
        "checkpoints": [
            {
                "name": "stand_midpoint",
                "relative_dir": "roboharness_run/trial_001/stand_midpoint_tick_0004",
                "cameras": ["track", "side", "top"],
            },
            {
                "name": "stand_end",
                "relative_dir": "roboharness_run/trial_001/stand_end_tick_0009/lag_3",
                "cameras": ["track", "side", "top"],
            },
            {
                "name": "peak_latency",
                "relative_dir": "roboharness_run/trial_001/peak_latency_tick_0011",
                "cameras": ["track", "side", "top"],
            },
        ],
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
            }
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


class ValidateSiteBundleTests(unittest.TestCase):
    def write_site(
        self,
        root: Path,
        *,
        proof_pack_manifest: dict[str, object],
        include_fallback_copy: bool = False,
        include_phase_sections: bool = False,
    ) -> None:
        (root / "assets" / "rerun-web-viewer").mkdir(parents=True)
        (root / "benchmarks" / "nvidia").mkdir(parents=True)
        policy_dir = root / "policies" / "gear_sonic"
        policy_dir.mkdir(parents=True)

        (root / "index.html").write_text(
            '<a href="benchmarks/nvidia/index.html">benchmarks</a>', encoding="utf-8"
        )
        (root / "assets" / "rerun-web-viewer" / "index.js").write_text(
            "export default {};",
            encoding="utf-8",
        )
        (root / "benchmarks" / "nvidia" / "index.html").write_text(
            "<html></html>",
            encoding="utf-8",
        )

        detail_copy = (
            "Screenshots unavailable for this build."
            if include_fallback_copy
            else "Visual checkpoints ready."
        )
        detail_sections = []
        if include_phase_sections:
            detail_sections.extend(
                [
                    '<section id="phase-timeline"></section>',
                    '<div id="phase-target-lag-selector" data-default-lag="0" data-selected-lag="0"></div>',
                    '<div id="phase-lag-selector" data-default-lag="3" data-selected-lag="3"></div>',
                    '<details class="phase-debug-panel" data-phase-debug-phase="stand"></details>',
                ]
            )
        (policy_dir / "index.html").write_text(
            "\n".join(
                [
                    '<script src="../../assets/rerun-web-viewer/index.js"></script>',
                    '<div data-rrd-file="run.rrd"></div>',
                    '<a href="proof_pack_manifest.json">Proof-pack manifest</a>',
                    detail_copy,
                    *detail_sections,
                ]
            ),
            encoding="utf-8",
        )

        (policy_dir / "proof_pack_manifest.json").write_text(
            json.dumps(proof_pack_manifest),
            encoding="utf-8",
        )

        for checkpoint in proof_pack_manifest.get("checkpoints", []):
            if not isinstance(checkpoint, dict):
                continue
            relative_dir = checkpoint.get("relative_dir")
            cameras = checkpoint.get("cameras")
            if isinstance(relative_dir, str) and isinstance(cameras, list):
                write_rgb_images(policy_dir, relative_dir, [str(camera) for camera in cameras])

        for checkpoint in proof_pack_manifest.get("phase_checkpoints", []):
            if not isinstance(checkpoint, dict):
                continue
            phase_kind = checkpoint.get("phase_kind")
            if phase_kind == "phase_end":
                lag_variants = checkpoint.get("lag_variants")
                target_lag_variants = checkpoint.get("target_lag_variants")
                if not isinstance(lag_variants, list):
                    continue
                for variant in lag_variants:
                    if not isinstance(variant, dict):
                        continue
                    relative_dir = variant.get("relative_dir")
                    cameras = variant.get("cameras")
                    if isinstance(relative_dir, str) and isinstance(cameras, list):
                        write_phase_end_variant_images(
                            policy_dir,
                            relative_dir,
                            [str(camera) for camera in cameras],
                        )
                if isinstance(target_lag_variants, list):
                    for variant in target_lag_variants:
                        if not isinstance(variant, dict):
                            continue
                        relative_dir = variant.get("relative_dir")
                        cameras = variant.get("cameras")
                        if isinstance(relative_dir, str) and isinstance(cameras, list):
                            write_rgb_images(
                                policy_dir,
                                relative_dir,
                                [str(camera) for camera in cameras],
                            )
                continue

            relative_dir = checkpoint.get("relative_dir")
            cameras = checkpoint.get("cameras")
            if isinstance(relative_dir, str) and isinstance(cameras, list):
                write_rgb_images(policy_dir, relative_dir, [str(camera) for camera in cameras])

        for checkpoint in proof_pack_manifest.get("diagnostic_checkpoints", []):
            if not isinstance(checkpoint, dict):
                continue
            relative_dir = checkpoint.get("relative_dir")
            cameras = checkpoint.get("cameras")
            if isinstance(relative_dir, str) and isinstance(cameras, list):
                write_rgb_images(policy_dir, relative_dir, [str(camera) for camera in cameras])

        (root / "manifest.json").write_text(
            json.dumps(
                [
                    {
                        "card_id": "gear_sonic",
                        "status": "ok",
                        "detail_page": "policies/gear_sonic/",
                        "_meta": {
                            "showcase_transport": "mujoco",
                            "proof_pack_manifest_file": "policies/gear_sonic/proof_pack_manifest.json",
                        },
                    }
                ]
            ),
            encoding="utf-8",
        )

    def test_validate_site_bundle_accepts_ok_proof_pack_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_site(root, proof_pack_manifest=make_generic_proof_pack_manifest())
            VALIDATOR.validate_site_bundle(root)

    def test_validate_site_bundle_rejects_skipped_proof_pack_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_site(
                root,
                proof_pack_manifest=make_generic_proof_pack_manifest(capture_status="skipped"),
                include_fallback_copy=True,
            )
            with self.assertRaises(SystemExit) as ctx:
                VALIDATOR.validate_site_bundle(root)
            self.assertIn("capture_status=ok", str(ctx.exception))

    def test_validate_site_bundle_accepts_phase_aware_proof_pack_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_site(
                root,
                proof_pack_manifest=make_phase_review_manifest(),
                include_phase_sections=True,
            )
            VALIDATOR.validate_site_bundle(root)

    def test_validate_site_bundle_rejects_phase_aware_manifest_missing_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = make_phase_review_manifest()
            manifest.pop("phase_checkpoints")
            self.write_site(
                root,
                proof_pack_manifest=manifest,
                include_phase_sections=True,
            )
            with self.assertRaises(SystemExit) as ctx:
                VALIDATOR.validate_site_bundle(root)
            self.assertIn("missing phase_checkpoints", str(ctx.exception))

    def test_validate_site_bundle_rejects_phase_aware_manifest_missing_lag_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_site(
                root,
                proof_pack_manifest=make_phase_review_manifest(),
                include_phase_sections=True,
            )
            missing_image = (
                root
                / "policies"
                / "gear_sonic"
                / "roboharness_run"
                / "trial_001"
                / "stand_end_tick_0009"
                / "lag_5"
                / "top_rgb.png"
            )
            missing_image.unlink()

            with self.assertRaises(SystemExit) as ctx:
                VALIDATOR.validate_site_bundle(root)
            self.assertIn("missing lag screenshot", str(ctx.exception))

    def test_validate_site_bundle_rejects_phase_aware_manifest_missing_target_lag_assets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_site(
                root,
                proof_pack_manifest=make_phase_review_manifest(),
                include_phase_sections=True,
            )
            missing_image = (
                root
                / "policies"
                / "gear_sonic"
                / "roboharness_run"
                / "trial_001"
                / "stand_end_tick_0009"
                / "target_lag_5"
                / "top_rgb.png"
            )
            missing_image.unlink()

            with self.assertRaises(SystemExit) as ctx:
                VALIDATOR.validate_site_bundle(root)
            self.assertIn("missing target lag screenshot", str(ctx.exception))

    def test_validate_site_bundle_rejects_phase_aware_path_escapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = make_phase_review_manifest()
            manifest["phase_checkpoints"][0]["relative_dir"] = "../escape"
            self.write_site(
                root,
                proof_pack_manifest=manifest,
                include_phase_sections=True,
            )

            with self.assertRaises(SystemExit) as ctx:
                VALIDATOR.validate_site_bundle(root)
            self.assertIn("escapes the site bundle root", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
