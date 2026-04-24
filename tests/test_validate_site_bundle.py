import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_site_bundle.py"

SPEC = importlib.util.spec_from_file_location("validate_site_bundle", VALIDATOR_PATH)
VALIDATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VALIDATOR)


class ValidateSiteBundleTests(unittest.TestCase):
    def write_site(
        self,
        root: Path,
        *,
        capture_status: str,
        include_fallback_copy: bool,
    ) -> None:
        (root / "assets" / "rerun-web-viewer").mkdir(parents=True)
        (root / "benchmarks" / "nvidia").mkdir(parents=True)
        policy_dir = root / "policies" / "gear_sonic"
        checkpoint_dir = policy_dir / "roboharness_run" / "trial_001" / "start_tick_0000"
        checkpoint_dir.mkdir(parents=True)

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
        (policy_dir / "index.html").write_text(
            "\n".join(
                [
                    '<script src="../../assets/rerun-web-viewer/index.js"></script>',
                    '<div data-rrd-file="run.rrd"></div>',
                    '<a href="proof_pack_manifest.json">Proof-pack manifest</a>',
                    detail_copy,
                ]
            ),
            encoding="utf-8",
        )

        checkpoints: list[dict[str, object]]
        if capture_status == "ok":
            checkpoints = [
                {
                    "name": "start_tick_0000",
                    "relative_dir": "roboharness_run/trial_001/start_tick_0000",
                    "cameras": ["track", "side", "top"],
                }
            ]
            for camera in ("track", "side", "top"):
                (checkpoint_dir / f"{camera}_rgb.png").write_bytes(b"png")
        else:
            checkpoints = []

        (policy_dir / "proof_pack_manifest.json").write_text(
            json.dumps(
                {
                    "capture_status": capture_status,
                    "checkpoints": checkpoints,
                }
            ),
            encoding="utf-8",
        )

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
            self.write_site(root, capture_status="ok", include_fallback_copy=False)
            VALIDATOR.validate_site_bundle(root)

    def test_validate_site_bundle_rejects_skipped_proof_pack_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self.write_site(root, capture_status="skipped", include_fallback_copy=True)
            with self.assertRaises(SystemExit) as ctx:
                VALIDATOR.validate_site_bundle(root)
            self.assertIn("capture_status=ok", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
