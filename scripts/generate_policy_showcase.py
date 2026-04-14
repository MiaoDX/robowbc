#!/usr/bin/env python3
"""Generate a mixed-source RoboWBC policy showcase as a static HTML artifact."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
from typing import Iterable

POLICIES = [
    {
        "id": "gear_sonic",
        "title": "GEAR-SONIC",
        "config": "configs/showcase/gear_sonic_real.toml",
        "source": "NVIDIA GR00T",
        "summary": "Real CPU planner_sonic.onnx run using the published multi-input velocity contract and the full Unitree G1 robot config.",
        "coverage": "Planner-only locomotion showcase",
        "execution_kind": "real",
        "checkpoint_source": "Published GEAR-SONIC ONNX checkpoints",
        "command_source": "runtime.velocity",
        "model_artifact": "models/gear-sonic/planner_sonic.onnx",
        "required_paths": [
            "models/gear-sonic/model_encoder.onnx",
            "models/gear-sonic/model_decoder.onnx",
            "models/gear-sonic/planner_sonic.onnx",
        ],
        "blocked_reason": "Requires downloaded GEAR-SONIC checkpoints. Run scripts/download_gear_sonic_models.sh or let CI warm the cache first.",
    },
    {
        "id": "decoupled_wbc",
        "title": "Decoupled WBC",
        "config": "configs/showcase/decoupled_wbc_real.toml",
        "source": "NVIDIA GR00T",
        "summary": "Real public GR00T WholeBodyControl run using the official 516D history contract plus separate balance and walk checkpoints.",
        "coverage": "Lower body RL + upper body default-pose baseline",
        "execution_kind": "real",
        "checkpoint_source": "Published GR00T WholeBodyControl ONNX checkpoints",
        "command_source": "runtime.velocity",
        "model_artifact": "models/decoupled-wbc/GR00T-WholeBodyControl-Walk.onnx",
        "required_paths": [
            "models/decoupled-wbc/GR00T-WholeBodyControl-Balance.onnx",
            "models/decoupled-wbc/GR00T-WholeBodyControl-Walk.onnx",
        ],
        "blocked_reason": "Requires downloaded GR00T WholeBodyControl checkpoints. Run scripts/download_decoupled_wbc_models.sh or let CI warm the cache first.",
    },
    {
        "id": "bfm_zero",
        "title": "BFM-Zero",
        "config": "configs/showcase/bfm_zero_mock.toml",
        "source": "CMU",
        "summary": "A single ONNX policy drives all joints from one velocity command.",
        "coverage": "Full-body velocity policy",
        "execution_kind": "fixture",
        "checkpoint_source": "Checked-in test ONNX fixture",
        "command_source": "runtime.velocity",
        "model_artifact": "crates/robowbc-ort/tests/fixtures/test_dynamic_identity.onnx",
    },
    {
        "id": "hover",
        "title": "HOVER",
        "config": "configs/showcase/hover_mock.toml",
        "source": "NVIDIA",
        "summary": "Masked multi-modal command space, running here in locomotion mode with the mock robot fixture.",
        "coverage": "Multi-modal masked policy",
        "execution_kind": "fixture",
        "checkpoint_source": "Checked-in test ONNX fixture",
        "command_source": "runtime.velocity",
        "model_artifact": "crates/robowbc-ort/tests/fixtures/test_dynamic_identity.onnx",
    },
    {
        "id": "wbc_agile",
        "title": "WBC-AGILE",
        "config": "configs/showcase/wbc_agile_real.toml",
        "source": "NVIDIA Isaac",
        "summary": "Real public G1 checkpoint using the published recurrent history tensors and lower-body target mapping.",
        "coverage": "Published G1 locomotion checkpoint",
        "execution_kind": "real",
        "checkpoint_source": "Published NVIDIA Isaac G1 ONNX checkpoint",
        "command_source": "runtime.velocity",
        "model_artifact": "models/wbc-agile/unitree_g1_velocity_e2e.onnx",
        "required_paths": [
            "models/wbc-agile/unitree_g1_velocity_e2e.onnx",
        ],
        "blocked_reason": "Requires downloaded WBC-AGILE G1 checkpoint. Run scripts/download_wbc_agile_models.sh or let CI warm the cache first.",
    },
]

NOT_YET_SHOWCASED = [
    {
        "name": "gear_sonic_tracking",
        "reason": "The published 1762D/994D encoder+decoder tracking contract is not integrated into the Rust runtime yet.",
    },
    {
        "name": "wholebody_vla",
        "reason": "The CLI now synthesizes KinematicPose commands, but no public WholeBodyVLA ONNX checkpoint is available yet.",
    },
    {
        "name": "py_model",
        "reason": "The showcase job is focused on compiled ORT-backed policies inside the Rust CLI.",
    },
]

COLORS = ["#0f766e", "#dc2626", "#2563eb", "#d97706", "#7c3aed", "#0891b2"]
RERUN_WEB_VIEWER_DIR = "_rerun_web_viewer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--robowbc-binary", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def resolve_ort_dylib(repo_root: Path) -> str | None:
    explicit = os.environ.get("ROBOWBC_ORT_DYLIB_PATH")
    if explicit:
        return explicit

    candidates = sorted(
        repo_root.glob(
            "target/debug/build/robowbc-ort-*/out/onnxruntime-linux-x64-1.24.2/lib/libonnxruntime.so.1.24.2"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        providers = path.parent / "libonnxruntime_providers_shared.so"
        if providers.exists():
            return str(path)
    return str(candidates[0]) if candidates else None


def resolve_rerun_web_viewer_version(repo_root: Path) -> str:
    lock_text = (repo_root / "Cargo.lock").read_text(encoding="utf-8")
    match = re.search(r'name = "rerun"\nversion = "([^"]+)"', lock_text)
    if match is None:
        raise SystemExit("failed to resolve rerun version from Cargo.lock")
    return match.group(1)


def vendor_rerun_web_viewer(repo_root: Path, output_dir: Path) -> dict[str, str]:
    version = resolve_rerun_web_viewer_version(repo_root)
    viewer_dir = output_dir / RERUN_WEB_VIEWER_DIR
    viewer_dir.mkdir(parents=True, exist_ok=True)
    version_file = viewer_dir / "VERSION"

    target_files = {
        "index_js": viewer_dir / "index.js",
        "viewer_js": viewer_dir / "re_viewer.js",
        "viewer_wasm": viewer_dir / "re_viewer_bg.wasm",
    }
    if (
        all(path.exists() for path in target_files.values())
        and version_file.exists()
        and version_file.read_text(encoding="utf-8").strip() == version
    ):
        return {
            "version": version,
            "module_path": f"./{RERUN_WEB_VIEWER_DIR}/index.js",
        }

    if shutil.which("npm") is None:
        raise SystemExit(
            "npm is required to vendor the embedded Rerun web viewer assets for the policy showcase"
        )

    with tempfile.TemporaryDirectory(prefix="robowbc-rerun-web-viewer-") as tempdir:
        temp_path = Path(tempdir)
        proc = subprocess.run(
            ["npm", "pack", f"@rerun-io/web-viewer@{version}"],
            cwd=temp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise SystemExit(
                "failed to fetch @rerun-io/web-viewer via npm pack:\n"
                f"{proc.stdout}\n--- STDERR ---\n{proc.stderr}"
            )

        tgz_files = list(temp_path.glob("rerun-io-web-viewer-*.tgz"))
        if len(tgz_files) != 1:
            raise SystemExit("unexpected npm pack output for @rerun-io/web-viewer")

        with tarfile.open(tgz_files[0]) as tar:
            try:
                tar.extractall(temp_path, filter="data")
            except TypeError:
                tar.extractall(temp_path)

        package_dir = temp_path / "package"
        index_text = (package_dir / "index.js").read_text(encoding="utf-8")
        index_text = index_text.replace('import("./re_viewer")', 'import("./re_viewer.js")')
        target_files["index_js"].write_text(index_text, encoding="utf-8")
        shutil.copy2(package_dir / "re_viewer.js", target_files["viewer_js"])
        shutil.copy2(package_dir / "re_viewer_bg.wasm", target_files["viewer_wasm"])
        version_file.write_text(version, encoding="utf-8")

    return {
        "version": version,
        "module_path": f"./{RERUN_WEB_VIEWER_DIR}/index.js",
    }


def append_report_and_vis(base_toml: str, policy_id: str, json_path: Path, rrd_path: Path) -> str:
    return (
        base_toml.rstrip()
        + f"\n\n[vis]\napp_id = \"robowbc-showcase-{policy_id}\"\nspawn_viewer = false\nsave_path = \"{rrd_path.as_posix()}\"\n\n[report]\noutput_path = \"{json_path.as_posix()}\"\nmax_frames = 120\n"
    )


def missing_required_paths(repo_root: Path, policy: dict[str, object]) -> list[str]:
    required = policy.get("required_paths", [])
    assert isinstance(required, list)
    missing: list[str] = []
    for rel_path in required:
        candidate = repo_root / str(rel_path)
        if not candidate.exists():
            missing.append(str(rel_path))
    return missing


def policy_meta(policy: dict[str, object], json_path: Path | None = None, rrd_path: Path | None = None, log_path: Path | None = None) -> dict[str, object]:
    meta = {
        "title": policy["title"],
        "source": policy["source"],
        "summary": policy["summary"],
        "coverage": policy["coverage"],
        "execution_kind": policy["execution_kind"],
        "checkpoint_source": policy["checkpoint_source"],
        "command_source": policy["command_source"],
        "model_artifact": policy.get("model_artifact", ""),
        "config_path": policy["config"],
        "required_paths": list(policy.get("required_paths", [])),
        "blocked_reason": policy.get("blocked_reason"),
    }
    if json_path is not None:
        meta["json_file"] = json_path.name
    if rrd_path is not None:
        meta["rrd_file"] = rrd_path.name
    if log_path is not None:
        meta["log_file"] = log_path.name
    return meta


def blocked_entry(repo_root: Path, policy: dict[str, object]) -> dict[str, object]:
    missing = missing_required_paths(repo_root, policy)
    return {
        "policy_name": policy["id"],
        "status": "blocked",
        "metrics": None,
        "frames": [],
        "joint_names": [],
        "command_kind": str(policy["command_source"]).removeprefix("runtime."),
        "command_data": [],
        "_meta": {
            **policy_meta(policy),
            "missing_paths": missing,
        },
    }


def run_policy(repo_root: Path, binary: Path, output_dir: Path, policy: dict[str, object], env: dict[str, str]) -> dict[str, object]:
    missing = missing_required_paths(repo_root, policy)
    if missing:
        return blocked_entry(repo_root, policy)

    policy_id = str(policy["id"])
    base_config = (repo_root / str(policy["config"])).read_text(encoding="utf-8")
    temp_config = output_dir / f"{policy_id}.toml"
    json_path = output_dir / f"{policy_id}.json"
    rrd_path = output_dir / f"{policy_id}.rrd"
    log_path = output_dir / f"{policy_id}.log"
    temp_config.write_text(
        append_report_and_vis(base_config, policy_id, json_path, rrd_path),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [str(binary), "run", "--config", str(temp_config)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    log_path.write_text(proc.stdout + "\n--- STDERR ---\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(
            f"policy showcase run failed for {policy_id} with exit code {proc.returncode}; see {log_path}"
        )

    report = json.loads(json_path.read_text(encoding="utf-8"))
    report["status"] = "ok"
    report["_meta"] = policy_meta(policy, json_path=json_path, rrd_path=rrd_path, log_path=log_path)
    return report


def series_from_frames(frames: list[dict[str, object]], field: str, joint_idx: int) -> list[float]:
    values: list[float] = []
    for frame in frames:
        data = frame.get(field, [])
        if isinstance(data, list) and joint_idx < len(data):
            values.append(float(data[joint_idx]))
    return values


def spark_svg(series_list: list[dict[str, object]], width: int = 360, height: int = 140) -> str:
    if not series_list:
        return '<svg class="chart" viewBox="0 0 360 140" role="img"></svg>'

    all_values = [value for series in series_list for value in series["values"]]
    if not all_values:
        return '<svg class="chart" viewBox="0 0 360 140" role="img"></svg>'

    min_v = min(all_values)
    max_v = max(all_values)
    if math.isclose(min_v, max_v):
        min_v -= 1.0
        max_v += 1.0

    pad = 12
    inner_w = width - 2 * pad
    inner_h = height - 2 * pad

    def point_x(idx: int, total: int) -> float:
        if total <= 1:
            return pad + inner_w / 2
        return pad + inner_w * idx / (total - 1)

    def point_y(val: float) -> float:
        return pad + inner_h * (1.0 - (val - min_v) / (max_v - min_v))

    paths = []
    for series in series_list:
        values = series["values"]
        pts = " ".join(
            f"{point_x(idx, len(values)):.2f},{point_y(val):.2f}"
            for idx, val in enumerate(values)
        )
        dash = ' stroke-dasharray="5 4"' if series.get("dashed") else ""
        paths.append(
            f'<polyline fill="none" stroke="{series["color"]}" stroke-width="2"{dash} points="{pts}" />'
        )

    baseline = point_y(0.0)
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img">'
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="14" />'
        f'<line x1="{pad}" y1="{baseline:.2f}" x2="{width-pad}" y2="{baseline:.2f}" class="baseline" />'
        + "".join(paths)
        + "</svg>"
    )


def format_vector(values: Iterable[float], limit: int = 6) -> str:
    items = list(values)
    head = ", ".join(f"{value:.3f}" for value in items[:limit])
    if len(items) > limit:
        head += ", ..."
    return f"[{head}]"


def pill(label: str, css_class: str) -> str:
    return f'<span class="pill {css_class}">{html.escape(label)}</span>'


def inline_recording_payload(entries: list[dict[str, object]], output_dir: Path) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for entry in entries:
        if entry.get("status") != "ok":
            continue
        meta = entry["_meta"]
        rrd_file = meta.get("rrd_file")
        if not isinstance(rrd_file, str) or not rrd_file:
            continue
        rrd_path = output_dir / rrd_file
        payload.append(
            {
                "policy_id": str(entry["policy_name"]),
                "title": str(meta["title"]),
                "rrd_file": rrd_file,
                "base64": base64.b64encode(rrd_path.read_bytes()).decode("ascii"),
            }
        )
    return payload


def render_html(entries: list[dict[str, object]], output_dir: Path, repo_root: Path) -> None:
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    sha = os.environ.get("GITHUB_SHA", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    commit_link = f"{server}/{repo}/commit/{sha}" if sha and repo else ""
    run_link = f"{server}/{repo}/actions/runs/{run_id}" if run_id and repo else ""
    viewer_assets = vendor_rerun_web_viewer(repo_root, output_dir)
    recordings_json = json.dumps(inline_recording_payload(entries, output_dir), separators=(",", ":"))

    overview_rows: list[str] = []
    cards: list[str] = []

    for entry in entries:
        meta = entry["_meta"]
        status = entry.get("status", "ok")
        execution_kind = str(meta["execution_kind"])
        status_html = pill("OK" if status == "ok" else "BLOCKED", "ok" if status == "ok" else "blocked")
        provenance_html = pill(execution_kind.upper(), execution_kind)

        metrics = entry.get("metrics") or {}
        ticks = metrics.get("ticks", "-")
        avg_inference = (
            f"{metrics['average_inference_ms']:.3f} ms" if metrics else "-"
        )
        achieved_hz = (
            f"{metrics['achieved_frequency_hz']:.2f} Hz" if metrics else "-"
        )
        dropped_frames = metrics.get("dropped_frames", "-")

        overview_rows.append(
            f"<tr><td><strong>{html.escape(str(meta['title']))}</strong><div class=\"muted\">{html.escape(str(entry['policy_name']))}</div></td>"
            f"<td>{status_html}</td>"
            f"<td>{provenance_html}</td>"
            f"<td>{html.escape(str(meta['coverage']))}</td>"
            f"<td>{ticks}</td>"
            f"<td>{avg_inference}</td>"
            f"<td>{achieved_hz}</td>"
            f"<td>{dropped_frames}</td></tr>"
        )

        badge_row = " ".join(
            [
                pill(execution_kind.upper(), execution_kind),
                pill(str(entry.get("command_kind", "")).upper(), "command"),
                pill(str(meta["command_source"]), "meta"),
            ]
        )

        if status != "ok":
            missing_paths = meta.get("missing_paths", [])
            missing_html = "<br />".join(f"<code>{html.escape(path)}</code>" for path in missing_paths)
            cards.append(
                f'''<section class="card blocked-card">
  <div class="card-header">
    <div>
      <h2>{html.escape(str(meta['title']))}</h2>
      <p class="muted">{html.escape(str(meta['source']))} · {html.escape(str(meta['coverage']))}</p>
    </div>
    <div class="badge-row">{badge_row}</div>
  </div>
  <p>{html.escape(str(meta['summary']))}</p>
  <div class="details-grid">
    <div>
      <span>Status</span>
      <strong>Blocked</strong>
    </div>
    <div>
      <span>Checkpoint source</span>
      <code>{html.escape(str(meta['checkpoint_source']))}</code>
    </div>
    <div>
      <span>Model artifact</span>
      <code>{html.escape(str(meta['model_artifact']))}</code>
    </div>
    <div>
      <span>Config</span>
      <code>{html.escape(str(meta['config_path']))}</code>
    </div>
  </div>
  <div class="blocked-reason">
    <strong>Why blocked:</strong> {html.escape(str(meta['blocked_reason']))}
  </div>
  <div class="blocked-paths">
    <span>Missing required paths</span>
    <div>{missing_html or '<span class="muted">None</span>'}</div>
  </div>
</section>'''
            )
            continue

        frames = entry.get("frames", [])
        metrics = entry["metrics"]
        joint_names = entry.get("joint_names", [])

        target_series = []
        for idx, joint_name in enumerate(joint_names[:4]):
            values = series_from_frames(frames, "target_positions", idx)
            target_series.append(
                {
                    "label": joint_name,
                    "values": values,
                    "color": COLORS[idx % len(COLORS)],
                }
            )

        actual_vs_target = []
        if joint_names:
            actual_vs_target = [
                {
                    "label": f"{joint_names[0]} actual",
                    "values": series_from_frames(frames, "actual_positions", 0),
                    "color": COLORS[0],
                    "dashed": True,
                },
                {
                    "label": f"{joint_names[0]} target",
                    "values": series_from_frames(frames, "target_positions", 0),
                    "color": COLORS[1],
                },
            ]

        latency_series = [
            {
                "label": "latency_ms",
                "values": [frame["inference_latency_ms"] for frame in frames],
                "color": COLORS[4],
            }
        ]

        cards.append(
            f'''<section class="card">
  <div class="card-header">
    <div>
      <h2>{html.escape(str(meta['title']))}</h2>
      <p class="muted">{html.escape(str(meta['source']))} · {html.escape(str(meta['coverage']))}</p>
    </div>
    <div class="badge-row">{badge_row}</div>
  </div>
  <p>{html.escape(str(meta['summary']))}</p>
  <div class="stats">
    <div><span>Robot</span><strong>{html.escape(str(entry['robot_name']))}</strong></div>
    <div><span>Ticks</span><strong>{metrics['ticks']}</strong></div>
    <div><span>Avg inference</span><strong>{metrics['average_inference_ms']:.3f} ms</strong></div>
    <div><span>Achieved rate</span><strong>{metrics['achieved_frequency_hz']:.2f} Hz</strong></div>
  </div>
  <div class="charts-grid">
    <figure>
      <figcaption>Target positions</figcaption>
      {spark_svg(target_series)}
    </figure>
    <figure>
      <figcaption>Joint 0 actual vs target</figcaption>
      {spark_svg(actual_vs_target)}
    </figure>
    <figure>
      <figcaption>Inference latency</figcaption>
      {spark_svg(latency_series)}
    </figure>
  </div>
  <div class="rerun-block">
    <div class="rerun-block-header">
      <strong>Embedded Rerun viewer</strong>
      <span class="muted">Loaded from the saved <code>{html.escape(str(meta['rrd_file']))}</code> recording.</span>
    </div>
    <div class="rerun-stage" data-rerun-policy="{html.escape(str(entry['policy_name']))}">
      <div class="rerun-stage-placeholder">
        <strong>Preparing interactive view</strong>
        <span>Mounts automatically when this card enters the viewport.</span>
      </div>
    </div>
  </div>
  <div class="details-grid">
    <div>
      <span>Command data</span>
      <code>{html.escape(format_vector(entry.get('command_data', [])))}</code>
    </div>
    <div>
      <span>Checkpoint source</span>
      <code>{html.escape(str(meta['checkpoint_source']))}</code>
    </div>
    <div>
      <span>Model artifact</span>
      <code>{html.escape(str(meta['model_artifact']))}</code>
    </div>
    <div>
      <span>Command source</span>
      <code>{html.escape(str(meta['command_source']))}</code>
    </div>
    <div>
      <span>First target frame</span>
      <code>{html.escape(format_vector(frames[0]['target_positions'] if frames else []))}</code>
    </div>
    <div>
      <span>Last target frame</span>
      <code>{html.escape(format_vector(frames[-1]['target_positions'] if frames else []))}</code>
    </div>
  </div>
  <p class="links"><a href="{html.escape(str(meta['rrd_file']))}">Rerun recording</a> · <a href="{html.escape(str(meta['json_file']))}">JSON summary</a> · <a href="{html.escape(str(meta['log_file']))}">run log</a> · <code>{html.escape(str(meta['config_path']))}</code></p>
</section>'''
        )

    excluded = "".join(
        f"<li><strong>{html.escape(item['name'])}</strong>: {html.escape(item['reason'])}</li>"
        for item in NOT_YET_SHOWCASED
    )
    commit_html = f'<a href="{html.escape(commit_link)}">{html.escape(sha[:12])}</a>' if commit_link else html.escape(sha[:12])
    run_html = f'<a href="{html.escape(run_link)}">Actions run</a>' if run_link else ""

    html_doc = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RoboWBC Policy Showcase</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #142033;
      --muted: #5f6f85;
      --border: #d9e0ea;
      --shadow: 0 18px 50px rgba(20, 32, 51, 0.08);
      --real-bg: #e7f7ef;
      --real-fg: #11643a;
      --fixture-bg: #e7f0ff;
      --fixture-fg: #1146a6;
      --blocked-bg: #fff1f2;
      --blocked-fg: #b42318;
      --command-bg: #fff6db;
      --command-fg: #8a5b00;
      --meta-bg: #edf2f7;
      --meta-fg: #334155;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; background: radial-gradient(circle at top, #eef7ff, var(--bg) 45%); color: var(--text); }}
    main {{ width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 40px 0 64px; }}
    h1, h2, h3, p {{ margin-top: 0; }}
    a {{ color: #0f5bd3; }}
    .hero {{ background: linear-gradient(135deg, #ffffff, #ecf4ff); border: 1px solid var(--border); border-radius: 28px; box-shadow: var(--shadow); padding: 32px; margin-bottom: 28px; }}
    .hero p {{ max-width: 80ch; line-height: 1.6; }}
    .meta-row {{ display: flex; gap: 16px; flex-wrap: wrap; color: var(--muted); font-size: 0.95rem; }}
    .overview, .footer-panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 24px; box-shadow: var(--shadow); padding: 24px; margin-bottom: 28px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 12px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }}
    th {{ font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
    .cards {{ display: grid; gap: 20px; }}
    .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 24px; box-shadow: var(--shadow); padding: 24px; }}
    .blocked-card {{ border-color: #f5c2c7; }}
    .card-header {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }}
    .badge-row {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }}
    .pill {{ border-radius: 999px; padding: 8px 12px; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; display: inline-flex; align-items: center; }}
    .pill.real {{ background: var(--real-bg); color: var(--real-fg); }}
    .pill.fixture {{ background: var(--fixture-bg); color: var(--fixture-fg); }}
    .pill.blocked {{ background: var(--blocked-bg); color: var(--blocked-fg); }}
    .pill.ok {{ background: var(--real-bg); color: var(--real-fg); }}
    .pill.command {{ background: var(--command-bg); color: var(--command-fg); }}
    .pill.meta {{ background: var(--meta-bg); color: var(--meta-fg); text-transform: none; }}
    .muted {{ color: var(--muted); }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 16px 0 20px; }}
    .stats div, .details-grid div {{ background: #f7f9fc; border: 1px solid var(--border); border-radius: 16px; padding: 12px 14px; }}
    .stats span, .details-grid span, .blocked-paths span {{ display: block; color: var(--muted); font-size: 0.82rem; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em; }}
    .stats strong {{ font-size: 1.05rem; }}
    .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 18px; }}
    figure {{ margin: 0; }}
    figcaption {{ margin-bottom: 10px; font-weight: 700; }}
    .chart {{ width: 100%; height: auto; display: block; }}
    .chart rect {{ fill: #fbfdff; stroke: var(--border); }}
    .chart .baseline {{ stroke: #d4dae3; stroke-width: 1; }}
    .details-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .rerun-block {{ margin: 0 0 18px; }}
    .rerun-block-header {{ display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-bottom: 10px; flex-wrap: wrap; }}
    .rerun-stage {{ min-height: 420px; border-radius: 18px; border: 1px solid var(--border); background: linear-gradient(180deg, #0f172a, #111827); overflow: hidden; position: relative; }}
    .rerun-stage canvas {{ display: block; width: 100%; height: 420px; }}
    .rerun-stage-placeholder, .rerun-stage-error {{ position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; padding: 20px; text-align: center; color: #e5edf8; font-size: 0.95rem; }}
    .rerun-stage-placeholder strong, .rerun-stage-error strong {{ color: #ffffff; }}
    .rerun-stage-error {{ background: linear-gradient(180deg, rgba(127, 29, 29, 0.95), rgba(69, 10, 10, 0.96)); }}
    .blocked-reason, .blocked-paths {{ margin-top: 18px; background: #fff7f7; border: 1px solid #f5c2c7; border-radius: 16px; padding: 14px; }}
    code {{ font-family: "IBM Plex Mono", "SFMono-Regular", monospace; font-size: 0.9rem; word-break: break-word; }}
    .links {{ margin-top: 16px; }}
    ul {{ margin-bottom: 0; }}
    @media (max-width: 720px) {{
      main {{ width: min(100% - 20px, 1180px); padding-top: 20px; }}
      .hero, .overview, .card, .footer-panel {{ padding: 20px; border-radius: 20px; }}
      .card-header {{ flex-direction: column; }}
      .badge-row {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>RoboWBC Policy Showcase</h1>
      <p>This artifact is generated automatically in CI from a mixed source set: fixture-backed policies that are always runnable in the repo, plus any real checkpoint integrations that are actually wired today. Missing real assets degrade to visible blocked cards instead of pretending the integration exists.</p>
      <p class="muted">Each successful card now embeds its saved Rerun recording directly in-page. The raw <code>.rrd</code> files are still available for download, and serving the folder over HTTP remains the most reliable way to open the interactive viewer locally.</p>
      <div class="meta-row">
        <span>Generated: {html.escape(generated_at)}</span>
        <span>Commit: {commit_html or 'local'}</span>
        <span>{run_html}</span>
      </div>
    </section>

    <section class="overview">
      <h2>Compared policies</h2>
      <p class="muted">Fixture cards use checked-in ONNX test models for repeatable CI coverage. Real cards use published checkpoints cached by CI and are only shown when the runtime path is actually executable.</p>
      <table>
        <thead>
          <tr><th>Policy</th><th>Status</th><th>Provenance</th><th>Coverage</th><th>Ticks</th><th>Avg inference</th><th>Achieved rate</th><th>Dropped frames</th></tr>
        </thead>
        <tbody>
          {''.join(overview_rows)}
        </tbody>
      </table>
    </section>

    <section class="cards">
      {''.join(cards)}
    </section>

    <section class="footer-panel">
      <h2>Not Yet In This Showcase</h2>
      <ul>{excluded}</ul>
    </section>
  </main>
  <script id="rerun-recordings" type="application/json">{recordings_json}</script>
  <script type="module">
    import {{ WebViewer }} from "{viewer_assets['module_path']}";

    const recordingData = JSON.parse(document.getElementById("rerun-recordings").textContent);
    const recordings = new Map(recordingData.map((item) => [item.policy_id, item]));
    const viewers = new Map();

    function base64ToUint8Array(base64String) {{
      const binary = atob(base64String);
      const bytes = new Uint8Array(binary.length);
      for (let idx = 0; idx < binary.length; idx += 1) {{
        bytes[idx] = binary.charCodeAt(idx);
      }}
      return bytes;
    }}

    function recordingBlobUrl(policyId) {{
      const recording = recordings.get(policyId);
      if (!recording) {{
        throw new Error(`missing embedded recording for ${{policyId}}`);
      }}
      if (!recording.blobUrl) {{
        const bytes = base64ToUint8Array(recording.base64);
        recording.blobUrl = URL.createObjectURL(new Blob([bytes], {{ type: "application/octet-stream" }}));
        delete recording.base64;
      }}
      return recording.blobUrl;
    }}

    function showStageError(stage, message) {{
      stage.replaceChildren();
      const errorNode = document.createElement("div");
      errorNode.className = "rerun-stage-error";
      const title = document.createElement("strong");
      title.textContent = "Viewer failed to start";
      const body = document.createElement("span");
      body.textContent = message;
      errorNode.append(title, body);
      stage.append(errorNode);
    }}

    async function mountStage(stage) {{
      const policyId = stage.dataset.rerunPolicy;
      if (!policyId || viewers.has(policyId)) {{
        return;
      }}

      stage.dataset.loading = "true";
      stage.replaceChildren();
      try {{
        const viewer = new WebViewer();
        await viewer.start(recordingBlobUrl(policyId), stage, {{
          width: "100%",
          height: "420px",
          hide_welcome_screen: true,
          theme: "light",
          render_backend: "webgl",
        }});
        viewers.set(policyId, viewer);
        stage.dataset.loading = "false";
      }} catch (error) {{
        console.error(`failed to mount rerun viewer for ${{policyId}}`, error);
        showStageError(stage, error instanceof Error ? error.message : String(error));
      }}
    }}

    const observer = new IntersectionObserver((entries) => {{
      for (const entry of entries) {{
        if (!entry.isIntersecting) {{
          continue;
        }}
        observer.unobserve(entry.target);
        void mountStage(entry.target);
      }}
    }}, {{ rootMargin: "180px 0px" }});

    const stages = [...document.querySelectorAll(".rerun-stage[data-rerun-policy]")];
    if (stages.length > 0) {{
      void mountStage(stages[0]);
    }}
    for (const stage of stages.slice(1)) {{
      observer.observe(stage);
    }}

    if (window.location.protocol === "file:") {{
      console.warn("Embedded Rerun viewers are most reliable when the showcase folder is served over HTTP.");
    }}
  </script>
</body>
</html>'''

    (output_dir / "index.html").write_text(html_doc, encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    binary = Path(args.robowbc_binary).resolve()
    if not binary.exists():
        raise SystemExit(f"robowbc binary not found: {binary}")

    env = os.environ.copy()
    dylib = resolve_ort_dylib(repo_root)
    if dylib:
        env.setdefault("ROBOWBC_ORT_DYLIB_PATH", dylib)

    entries = [run_policy(repo_root, binary, output_dir, policy, env) for policy in POLICIES]
    render_html(entries, output_dir, repo_root)
    print(f"wrote showcase report to {output_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
