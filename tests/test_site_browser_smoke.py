import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE_PATH = ROOT / "scripts" / "site" / "site_browser_smoke.py"

SPEC = importlib.util.spec_from_file_location("site_browser_smoke", SMOKE_PATH)
SMOKE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SMOKE)


def make_probe_result() -> dict[str, object]:
    return {
        "policy": "gear_sonic",
        "plan": {
            "initial_target_lag": 0,
            "initial_actual_lag": 3,
            "target_probe_lag": 2,
            "actual_probe_lag": 1,
            "reset_target_lag": 0,
            "final_actual_lag": 5,
        },
        "results": [
            {
                "tag": "initial",
                "label": "T+0 (0 ms) · A+3 (60 ms)",
                "srcKind": "asset",
                "selectedTarget": "0",
                "selectedActual": "3",
            },
            {
                "tag": "target-2",
                "label": "T+2 (40 ms) · A+3 (60 ms)",
                "srcKind": "data-url",
                "selectedTarget": "2",
                "selectedActual": "3",
            },
            {
                "tag": "actual-1",
                "label": "T+2 (40 ms) · A+1 (20 ms)",
                "srcKind": "data-url",
                "selectedTarget": "2",
                "selectedActual": "1",
            },
            {
                "tag": "target-reset",
                "label": "T+0 (0 ms) · A+1 (20 ms)",
                "srcKind": "asset",
                "selectedTarget": "0",
                "selectedActual": "1",
            },
            {
                "tag": "actual-5",
                "label": "T+0 (0 ms) · A+5 (100 ms)",
                "srcKind": "asset",
                "selectedTarget": "0",
                "selectedActual": "5",
            },
        ],
        "sampleLabels": [
            "T+0 (0 ms) · A+5 (100 ms)",
            "T+0 (0 ms) · A+5 (100 ms)",
            "T+0 (0 ms) · A+5 (100 ms)",
        ],
    }


class SiteBrowserSmokeTests(unittest.TestCase):
    def test_build_probe_html_targets_policy_page(self) -> None:
        html_text = SMOKE.build_probe_html("gear_sonic")

        self.assertIn("/policies/gear_sonic/", html_text)
        self.assertIn('id="smoke-result"', html_text)
        self.assertIn('id="policy-frame"', html_text)

    def test_extract_probe_result_reads_json_from_dumped_dom(self) -> None:
        dumped_dom = (
            "<html><body><pre id=\"smoke-result\">"
            "{&quot;policy&quot;: &quot;gear_sonic&quot;, &quot;plan&quot;: {}}"
            "</pre></body></html>"
        )

        result = SMOKE.extract_probe_result(dumped_dom)

        self.assertEqual(result["policy"], "gear_sonic")
        self.assertEqual(result["plan"], {})

    def test_validate_probe_result_accepts_expected_transitions(self) -> None:
        SMOKE.validate_probe_result(make_probe_result(), "gear_sonic")

    def test_validate_probe_result_rejects_wrong_transition(self) -> None:
        result = make_probe_result()
        result["results"][1]["srcKind"] = "asset"

        with self.assertRaises(SystemExit):
            SMOKE.validate_probe_result(result, "gear_sonic")


if __name__ == "__main__":
    unittest.main()
