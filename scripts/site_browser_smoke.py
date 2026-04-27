#!/usr/bin/env python3
"""Run a reusable headless browser smoke check against a built RoboWBC site."""

from __future__ import annotations

import argparse
import contextlib
import html
import http.server
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

PROBE_RESULT_PATTERN = re.compile(
    r'<pre id="smoke-result"[^>]*>(?P<payload>.*?)</pre>',
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory of the generated site bundle",
    )
    parser.add_argument(
        "--policy",
        default="gear_sonic",
        help="Policy detail page to inspect under policies/<policy>/",
    )
    parser.add_argument(
        "--bind",
        default="127.0.0.1",
        help="Bind address for the temporary HTTP server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port for the temporary HTTP server; use 0 for an ephemeral port",
    )
    parser.add_argument(
        "--chrome-binary",
        default="",
        help="Optional path to the Chrome/Chromium binary to use",
    )
    parser.add_argument(
        "--timeout-secs",
        type=int,
        default=30,
        help="Timeout for the browser probe command",
    )
    parser.add_argument(
        "--keep-probe",
        action="store_true",
        help="Keep the generated probe HTML file in the site root for manual debugging",
    )
    return parser.parse_args()


def validate_policy_name(policy: str) -> str:
    candidate = Path(policy)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise SystemExit(f"policy id must stay within policies/<id>/, got {policy!r}")
    if len(candidate.parts) != 1:
        raise SystemExit(f"policy id must be a single path component, got {policy!r}")
    return candidate.parts[0]


def find_chrome_binary(requested: str) -> str:
    if requested:
        resolved = shutil.which(requested) or requested
        if Path(resolved).is_file():
            return str(Path(resolved))
        raise SystemExit(f"chrome binary not found: {requested}")

    for candidate in (
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
    ):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise SystemExit(
        "could not find a Chrome/Chromium binary; pass --chrome-binary explicitly"
    )


def build_probe_html(policy: str) -> str:
    policy_path = f"/policies/{policy}/"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>RoboWBC Site Browser Smoke</title>
  <style>
    body {{ font-family: "IBM Plex Sans", "Segoe UI", sans-serif; margin: 24px; color: #142033; }}
    iframe {{ width: 100%; min-height: 960px; border: 1px solid #d9e0ea; border-radius: 16px; }}
    pre {{ margin-top: 20px; padding: 16px; border-radius: 14px; background: #0f172a; color: #e2e8f0; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>RoboWBC Site Browser Smoke</h1>
  <p>Probing <code>{html.escape(policy_path)}</code>.</p>
  <iframe id="policy-frame" src="about:blank" title="policy detail page"></iframe>
  <pre id="smoke-result">{{"status": "pending"}}</pre>
  <script>
    const POLICY_PATH = {json.dumps(policy_path)};

    function writeResult(payload) {{
      document.getElementById("smoke-result").textContent = JSON.stringify(payload, null, 2);
      document.body.dataset.status = payload.error ? "error" : "ok";
    }}

    function sleep(ms) {{
      return new Promise((resolve) => window.setTimeout(resolve, ms));
    }}

    async function waitFor(predicate, label, timeoutMs = 15000) {{
      const startedAt = Date.now();
      while ((Date.now() - startedAt) < timeoutMs) {{
        const value = await predicate();
        if (value) {{
          return value;
        }}
        await sleep(100);
      }}
      throw new Error(`timed out waiting for ${{label}}`);
    }}

    function findSelectedLag(selector) {{
      if (!selector) {{
        return null;
      }}
      const active = [...selector.querySelectorAll(".phase-lag-button[data-lag]")].find((button) => (
        button.dataset.active === "true" || button.getAttribute("aria-pressed") === "true"
      ));
      return active ? active.dataset.lag : null;
    }}

    function snapshot(doc, tag) {{
      const image = doc.querySelector("img[data-phase-lag-image]");
      const label = doc.querySelector("[data-phase-lag-label]")?.textContent?.replace(/\\s+/g, " ").trim() ?? null;
      const targetSelector = doc.getElementById("phase-target-lag-selector");
      const actualSelector = doc.getElementById("phase-lag-selector");
      const imageSrc = image?.getAttribute("src") ?? null;
      return {{
        tag,
        label,
        srcKind: imageSrc ? (imageSrc.startsWith("data:image/") ? "data-url" : "asset") : null,
        srcTail: imageSrc ? imageSrc.slice(-80) : null,
        selectedTarget: findSelectedLag(targetSelector),
        selectedActual: findSelectedLag(actualSelector),
      }};
    }}

    function selectorLags(selector) {{
      return [...selector.querySelectorAll(".phase-lag-button[data-lag]")]
        .map((button) => Number.parseInt(button.dataset.lag || "", 10))
        .filter((value) => Number.isFinite(value))
        .sort((left, right) => left - right);
    }}

    function chooseLag(lags, preferred, current) {{
      if (lags.includes(preferred) && preferred !== current) {{
        return preferred;
      }}
      const positive = lags.find((lag) => lag > 0 && lag !== current);
      if (positive !== undefined) {{
        return positive;
      }}
      const alternate = lags.find((lag) => lag !== current);
      if (alternate !== undefined) {{
        return alternate;
      }}
      return current;
    }}

    async function run() {{
      const iframe = document.getElementById("policy-frame");
      iframe.src = POLICY_PATH;
      await new Promise((resolve, reject) => {{
        const timeout = window.setTimeout(() => reject(new Error("iframe load timed out")), 15000);
        iframe.addEventListener("load", () => {{
          window.clearTimeout(timeout);
          resolve();
        }}, {{ once: true }});
      }});

      const doc = iframe.contentDocument;
      if (!doc) {{
        throw new Error("iframe contentDocument unavailable");
      }}
      await waitFor(
        () => (
          doc.getElementById("phase-target-lag-selector") &&
          doc.getElementById("phase-lag-selector") &&
          doc.querySelector("img[data-phase-lag-image]")
        ),
        "phase review controls",
      );

      const targetSelector = doc.getElementById("phase-target-lag-selector");
      const actualSelector = doc.getElementById("phase-lag-selector");
      if (!targetSelector || !actualSelector) {{
        throw new Error("missing phase lag selectors");
      }}

      const initialTargetLag = Number.parseInt(
        targetSelector.dataset.selectedLag || targetSelector.dataset.defaultLag || "0",
        10,
      );
      const initialActualLag = Number.parseInt(
        actualSelector.dataset.selectedLag || actualSelector.dataset.defaultLag || "0",
        10,
      );
      const targetLags = selectorLags(targetSelector);
      const actualLags = selectorLags(actualSelector);
      const targetProbeLag = chooseLag(targetLags, 2, initialTargetLag);
      const actualProbeLag = chooseLag(actualLags, 1, initialActualLag);
      const resetTargetLag = targetLags.includes(0) ? 0 : initialTargetLag;
      const finalActualLag = chooseLag(actualLags, 5, actualProbeLag);

      const plan = {{
        initial_target_lag: initialTargetLag,
        initial_actual_lag: initialActualLag,
        target_probe_lag: targetProbeLag,
        actual_probe_lag: actualProbeLag,
        reset_target_lag: resetTargetLag,
        final_actual_lag: finalActualLag,
      }};
      if (targetProbeLag === initialTargetLag || targetProbeLag === 0) {{
        throw new Error("target probe lag must move away from the default and stay above zero");
      }}
      if (actualProbeLag === initialActualLag) {{
        throw new Error("actual probe lag must move away from the default");
      }}
      if (finalActualLag === actualProbeLag) {{
        throw new Error("final actual lag must differ from the probe lag");
      }}

      const results = [];
      results.push(snapshot(doc, "initial"));

      const targetProbeButton = targetSelector.querySelector(`.phase-lag-button[data-lag="${{targetProbeLag}}"]`);
      if (!targetProbeButton) {{
        throw new Error(`missing target probe button for lag ${{targetProbeLag}}`);
      }}
      targetProbeButton.click();
      await waitFor(() => {{
        const state = snapshot(doc, "target-2");
        return (
          state.selectedTarget === String(targetProbeLag) &&
          state.selectedActual === String(initialActualLag) &&
          state.srcKind === "data-url" &&
          state.label?.includes(`T+${{targetProbeLag}}`) &&
          state.label?.includes(`A+${{initialActualLag}}`)
        );
      }}, "target probe update");
      results.push(snapshot(doc, "target-2"));

      const actualProbeButton = actualSelector.querySelector(`.phase-lag-button[data-lag="${{actualProbeLag}}"]`);
      if (!actualProbeButton) {{
        throw new Error(`missing actual probe button for lag ${{actualProbeLag}}`);
      }}
      actualProbeButton.click();
      await waitFor(() => {{
        const state = snapshot(doc, "actual-1");
        return (
          state.selectedTarget === String(targetProbeLag) &&
          state.selectedActual === String(actualProbeLag) &&
          state.srcKind === "data-url" &&
          state.label?.includes(`T+${{targetProbeLag}}`) &&
          state.label?.includes(`A+${{actualProbeLag}}`)
        );
      }}, "actual probe update");
      results.push(snapshot(doc, "actual-1"));

      const targetResetButton = targetSelector.querySelector(`.phase-lag-button[data-lag="${{resetTargetLag}}"]`);
      if (!targetResetButton) {{
        throw new Error(`missing target reset button for lag ${{resetTargetLag}}`);
      }}
      targetResetButton.click();
      await waitFor(() => {{
        const state = snapshot(doc, "target-reset");
        return (
          state.selectedTarget === String(resetTargetLag) &&
          state.selectedActual === String(actualProbeLag) &&
          state.srcKind === "asset" &&
          state.label?.includes(`T+${{resetTargetLag}}`) &&
          state.label?.includes(`A+${{actualProbeLag}}`)
        );
      }}, "target reset update");
      results.push(snapshot(doc, "target-reset"));

      const finalActualButton = actualSelector.querySelector(`.phase-lag-button[data-lag="${{finalActualLag}}"]`);
      if (!finalActualButton) {{
        throw new Error(`missing final actual button for lag ${{finalActualLag}}`);
      }}
      finalActualButton.click();
      await waitFor(() => {{
        const state = snapshot(doc, "actual-5");
        return (
          state.selectedTarget === String(resetTargetLag) &&
          state.selectedActual === String(finalActualLag) &&
          state.srcKind === "asset" &&
          state.label?.includes(`T+${{resetTargetLag}}`) &&
          state.label?.includes(`A+${{finalActualLag}}`)
        );
      }}, "final actual update");
      results.push(snapshot(doc, "actual-5"));

      const sampleLabels = [...doc.querySelectorAll("[data-phase-lag-label]")]
        .slice(0, 3)
        .map((node) => node.textContent.replace(/\\s+/g, " ").trim());

      writeResult({{
        policy: {json.dumps(policy)},
        plan,
        results,
        sampleLabels,
      }});
    }}

    void run().catch((error) => {{
      writeResult({{
        policy: {json.dumps(policy)},
        error: error instanceof Error ? error.stack || error.message : String(error),
      }});
    }});
  </script>
</body>
</html>
"""


def extract_probe_result(dumped_dom: str) -> dict[str, object]:
    match = PROBE_RESULT_PATTERN.search(dumped_dom)
    if match is None:
        raise SystemExit("browser probe output missing #smoke-result payload")
    payload_text = html.unescape(match.group("payload")).strip()
    if not payload_text:
        raise SystemExit("browser probe emitted an empty result payload")
    payload = json.loads(payload_text)
    if not isinstance(payload, dict):
        raise SystemExit("browser probe payload must be a JSON object")
    return payload


def validate_probe_result(result: dict[str, object], policy: str) -> None:
    if result.get("policy") != policy:
        raise SystemExit(
            f"browser probe returned result for {result.get('policy')!r}, expected {policy!r}"
        )
    if result.get("error"):
        raise SystemExit(f"browser probe failed: {result['error']}")

    plan = result.get("plan")
    if not isinstance(plan, dict):
        raise SystemExit("browser probe result missing plan payload")
    expected_plan_keys = {
        "initial_target_lag",
        "initial_actual_lag",
        "target_probe_lag",
        "actual_probe_lag",
        "reset_target_lag",
        "final_actual_lag",
    }
    if not expected_plan_keys.issubset(plan):
        raise SystemExit(f"browser probe plan is incomplete: {sorted(plan.keys())}")

    results = result.get("results")
    if not isinstance(results, list):
        raise SystemExit("browser probe result missing transition snapshots")
    snapshot_map = {
        snapshot.get("tag"): snapshot
        for snapshot in results
        if isinstance(snapshot, dict) and isinstance(snapshot.get("tag"), str)
    }
    expected_snapshots = {
        "initial": {
            "srcKind": "asset",
            "selectedTarget": str(plan["initial_target_lag"]),
            "selectedActual": str(plan["initial_actual_lag"]),
        },
        "target-2": {
            "srcKind": "data-url",
            "selectedTarget": str(plan["target_probe_lag"]),
            "selectedActual": str(plan["initial_actual_lag"]),
        },
        "actual-1": {
            "srcKind": "data-url",
            "selectedTarget": str(plan["target_probe_lag"]),
            "selectedActual": str(plan["actual_probe_lag"]),
        },
        "target-reset": {
            "srcKind": "asset",
            "selectedTarget": str(plan["reset_target_lag"]),
            "selectedActual": str(plan["actual_probe_lag"]),
        },
        "actual-5": {
            "srcKind": "asset",
            "selectedTarget": str(plan["reset_target_lag"]),
            "selectedActual": str(plan["final_actual_lag"]),
        },
    }

    for tag, expected in expected_snapshots.items():
        snapshot = snapshot_map.get(tag)
        if not isinstance(snapshot, dict):
            raise SystemExit(f"browser probe missing snapshot {tag!r}")
        if snapshot.get("srcKind") != expected["srcKind"]:
            raise SystemExit(
                f"browser probe snapshot {tag!r} expected srcKind={expected['srcKind']!r}, "
                f"found {snapshot.get('srcKind')!r}"
            )
        if snapshot.get("selectedTarget") != expected["selectedTarget"]:
            raise SystemExit(
                f"browser probe snapshot {tag!r} expected selectedTarget={expected['selectedTarget']!r}, "
                f"found {snapshot.get('selectedTarget')!r}"
            )
        if snapshot.get("selectedActual") != expected["selectedActual"]:
            raise SystemExit(
                f"browser probe snapshot {tag!r} expected selectedActual={expected['selectedActual']!r}, "
                f"found {snapshot.get('selectedActual')!r}"
            )
        label = snapshot.get("label")
        if not isinstance(label, str):
            raise SystemExit(f"browser probe snapshot {tag!r} missing label text")
        if f"T+{expected['selectedTarget']}" not in label:
            raise SystemExit(
                f"browser probe snapshot {tag!r} label missing target lag {expected['selectedTarget']!r}: {label!r}"
            )
        if f"A+{expected['selectedActual']}" not in label:
            raise SystemExit(
                f"browser probe snapshot {tag!r} label missing actual lag {expected['selectedActual']!r}: {label!r}"
            )

    sample_labels = result.get("sampleLabels")
    if not isinstance(sample_labels, list) or not sample_labels:
        raise SystemExit("browser probe missing sample label coverage")
    final_label = snapshot_map["actual-5"]["label"]
    if not all(label == final_label for label in sample_labels if isinstance(label, str)):
        raise SystemExit(
            "browser probe sample labels did not converge on the final phase lag label"
        )


def wait_for_url(url: str, timeout_secs: float) -> None:
    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if 200 <= response.status < 500:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)
    raise SystemExit(f"temporary site server did not become ready: {url}")


@contextlib.contextmanager
def serve_directory(root: Path, bind: str, port: int) -> tuple[str, int]:
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, format: str, *args) -> None:
            return

    server = http.server.ThreadingHTTPServer((bind, port), QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield bind, int(server.server_address[1])
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def run_browser_probe(chrome_binary: str, url: str, timeout_secs: int) -> str:
    command = [
        chrome_binary,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--run-all-compositor-stages-before-draw",
        f"--virtual-time-budget={timeout_secs * 1000}",
        "--dump-dom",
        url,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_secs,
            env={**os.environ, "HOME": os.environ.get("HOME", str(Path.home()))},
        )
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            f"chrome browser probe timed out after {timeout_secs}s while loading {url}"
        ) from exc
    if completed.returncode != 0:
        raise SystemExit(
            "chrome browser probe failed with "
            f"exit code {completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"site root does not exist: {root}")

    policy = validate_policy_name(args.policy)
    detail_page = root / "policies" / policy / "index.html"
    if not detail_page.is_file():
        raise SystemExit(f"missing policy detail page: {detail_page}")

    chrome_binary = find_chrome_binary(args.chrome_binary)
    probe_name = f".site-browser-smoke-{uuid.uuid4().hex}.html"
    probe_path = root / probe_name
    probe_path.write_text(build_probe_html(policy), encoding="utf-8")

    try:
        with serve_directory(root, args.bind, args.port) as (bind, port):
            probe_url = f"http://{bind}:{port}/{probe_name}"
            wait_for_url(probe_url, timeout_secs=5.0)
            dumped_dom = run_browser_probe(
                chrome_binary=chrome_binary,
                url=probe_url,
                timeout_secs=args.timeout_secs,
            )
        result = extract_probe_result(dumped_dom)
        validate_probe_result(result, policy)
    finally:
        if not args.keep_probe and probe_path.exists():
            probe_path.unlink()

    final_snapshot = next(
        snapshot
        for snapshot in result["results"]
        if isinstance(snapshot, dict) and snapshot.get("tag") == "actual-5"
    )
    print(
        "site browser smoke check passed: "
        f"policy={policy} final_label={final_snapshot['label']} src_kind={final_snapshot['srcKind']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
