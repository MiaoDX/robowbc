#!/usr/bin/env python3
"""Generate a fixture-backed RoboWBC policy showcase as a static HTML artifact."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Iterable

POLICIES = [
    {
        "id": "decoupled_wbc",
        "title": "Decoupled WBC",
        "config": "configs/showcase/decoupled_wbc_mock.toml",
        "source": "NVIDIA GR00T",
        "summary": "Lower-body RL outputs targets for joints 0-1 while the upper body holds the default pose.",
        "coverage": "Lower body RL + upper body baseline",
        "fixture_model": "test_dynamic_identity.onnx",
    },
    {
        "id": "bfm_zero",
        "title": "BFM-Zero",
        "config": "configs/showcase/bfm_zero_mock.toml",
        "source": "CMU",
        "summary": "A single ONNX policy drives all joints from one velocity command.",
        "coverage": "Full-body velocity policy",
        "fixture_model": "test_dynamic_identity.onnx",
    },
    {
        "id": "hover",
        "title": "HOVER",
        "config": "configs/showcase/hover_mock.toml",
        "source": "NVIDIA",
        "summary": "Masked multi-modal command space, running here in locomotion mode with the mock robot fixture.",
        "coverage": "Multi-modal masked policy",
        "fixture_model": "test_dynamic_identity.onnx",
    },
    {
        "id": "wbc_agile",
        "title": "WBC-AGILE",
        "config": "configs/showcase/wbc_agile_mock.toml",
        "source": "NVIDIA Isaac",
        "summary": "A single whole-body policy over all actuated joints, using the same mock fixture for CI reproducibility.",
        "coverage": "Full-body velocity policy",
        "fixture_model": "test_dynamic_identity.onnx",
    },
]

EXCLUDED_POLICIES = [
    {
        "name": "gear_sonic",
        "reason": "The current fixture set does not include a compatible three-stage encoder/planner/decoder mock pipeline.",
    },
    {
        "name": "wholebody_vla",
        "reason": "The current showcase flow does not synthesize KinematicPose commands for an end-to-end CLI run.",
    },
    {
        "name": "py_model",
        "reason": "The showcase job is focused on no-download ORT-backed policies compiled into the Rust CLI.",
    },
]

COLORS = ["#0f766e", "#dc2626", "#2563eb", "#d97706", "#7c3aed", "#0891b2"]


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


def append_report_and_vis(base_toml: str, policy_id: str, json_path: Path, rrd_path: Path) -> str:
    return (
        base_toml.rstrip()
        + f"\n\n[vis]\napp_id = \"robowbc-showcase-{policy_id}\"\nspawn_viewer = false\nsave_path = \"{rrd_path.as_posix()}\"\n\n[report]\noutput_path = \"{json_path.as_posix()}\"\nmax_frames = 120\n"
    )


def run_policy(repo_root: Path, binary: Path, output_dir: Path, policy: dict[str, str], env: dict[str, str]) -> dict:
    policy_id = policy["id"]
    base_config = (repo_root / policy["config"]).read_text(encoding="utf-8")
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
    report["_meta"] = {
        "title": policy["title"],
        "source": policy["source"],
        "summary": policy["summary"],
        "coverage": policy["coverage"],
        "fixture_model": policy["fixture_model"],
        "config_path": policy["config"],
        "json_file": json_path.name,
        "rrd_file": rrd_path.name,
        "log_file": log_path.name,
    }
    return report


def series_from_frames(frames: list[dict], field: str, joint_idx: int) -> list[float]:
    values: list[float] = []
    for frame in frames:
        data = frame.get(field, [])
        if joint_idx < len(data):
            values.append(float(data[joint_idx]))
    return values


def spark_svg(series_list: list[dict], width: int = 360, height: int = 140) -> str:
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
        + '</svg>'
    )


def format_vector(values: Iterable[float], limit: int = 6) -> str:
    items = list(values)
    head = ", ".join(f"{value:.3f}" for value in items[:limit])
    if len(items) > limit:
        head += ", ..."
    return f"[{head}]"


def render_html(reports: list[dict], output_dir: Path) -> None:
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    sha = os.environ.get("GITHUB_SHA", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    commit_link = f"{server}/{repo}/commit/{sha}" if sha and repo else ""
    run_link = f"{server}/{repo}/actions/runs/{run_id}" if run_id and repo else ""

    overview_rows = []
    cards = []
    manifest = []

    for report in reports:
        meta = report["_meta"]
        frames = report.get("frames", [])
        metrics = report["metrics"]
        joint_names = report.get("joint_names", [])

        target_series = []
        for idx, joint_name in enumerate(joint_names[:4]):
            values = series_from_frames(frames, "target_positions", idx)
            target_series.append({
                "label": joint_name,
                "values": values,
                "color": COLORS[idx % len(COLORS)],
            })

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

        overview_rows.append(
            f"<tr><td><strong>{html.escape(meta['title'])}</strong><div class=\"muted\">{html.escape(report['policy_name'])}</div></td>"
            f"<td>{html.escape(meta['coverage'])}</td>"
            f"<td>{metrics['ticks']}</td>"
            f"<td>{metrics['average_inference_ms']:.3f} ms</td>"
            f"<td>{metrics['achieved_frequency_hz']:.2f} Hz</td>"
            f"<td>{metrics['dropped_frames']}</td></tr>"
        )

        cards.append(f"""<section class=\"card\">\n  <div class=\"card-header\">\n    <div>\n      <h2>{html.escape(meta['title'])}</h2>\n      <p class=\"muted\">{html.escape(meta['source'])} · {html.escape(meta['coverage'])}</p>\n    </div>\n    <div class=\"pill\">{html.escape(report['command_kind'])}</div>\n  </div>\n  <p>{html.escape(meta['summary'])}</p>\n  <div class=\"stats\">\n    <div><span>Robot</span><strong>{html.escape(report['robot_name'])}</strong></div>\n    <div><span>Ticks</span><strong>{metrics['ticks']}</strong></div>\n    <div><span>Avg inference</span><strong>{metrics['average_inference_ms']:.3f} ms</strong></div>\n    <div><span>Achieved rate</span><strong>{metrics['achieved_frequency_hz']:.2f} Hz</strong></div>\n  </div>\n  <div class=\"charts-grid\">\n    <figure>\n      <figcaption>Target positions</figcaption>\n      {spark_svg(target_series)}\n    </figure>\n    <figure>\n      <figcaption>Joint 0 actual vs target</figcaption>\n      {spark_svg(actual_vs_target)}\n    </figure>\n    <figure>\n      <figcaption>Inference latency</figcaption>\n      {spark_svg(latency_series)}\n    </figure>\n  </div>\n  <div class=\"details-grid\">\n    <div>\n      <span>Command data</span>\n      <code>{html.escape(format_vector(report.get('command_data', [])))}</code>\n    </div>\n    <div>\n      <span>First target frame</span>\n      <code>{html.escape(format_vector(frames[0]['target_positions'] if frames else []))}</code>\n    </div>\n    <div>\n      <span>Last target frame</span>\n      <code>{html.escape(format_vector(frames[-1]['target_positions'] if frames else []))}</code>\n    </div>\n    <div>\n      <span>Fixture model</span>\n      <code>{html.escape(meta['fixture_model'])}</code>\n    </div>\n  </div>\n  <p class=\"links\"><a href=\"{html.escape(meta['rrd_file'])}\">Rerun recording</a> · <a href=\"{html.escape(meta['json_file'])}\">JSON summary</a> · <a href=\"{html.escape(meta['log_file'])}\">run log</a> · <code>{html.escape(meta['config_path'])}</code></p>\n</section>""")
        manifest.append(report)

    excluded = "".join(
        f"<li><strong>{html.escape(item['name'])}</strong>: {html.escape(item['reason'])}</li>"
        for item in EXCLUDED_POLICIES
    )
    commit_html = f'<a href="{html.escape(commit_link)}">{html.escape(sha[:12])}</a>' if commit_link else html.escape(sha[:12])
    run_html = f'<a href="{html.escape(run_link)}">Actions run</a>' if run_link else ''

    html_doc = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
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
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: \"IBM Plex Sans\", \"Segoe UI\", sans-serif; background: radial-gradient(circle at top, #eef7ff, var(--bg) 45%); color: var(--text); }}
    main {{ width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 40px 0 64px; }}
    h1, h2, h3, p {{ margin-top: 0; }}
    a {{ color: #0f5bd3; }}
    .hero {{ background: linear-gradient(135deg, #ffffff, #ecf4ff); border: 1px solid var(--border); border-radius: 28px; box-shadow: var(--shadow); padding: 32px; margin-bottom: 28px; }}
    .hero p {{ max-width: 76ch; line-height: 1.6; }}
    .meta-row {{ display: flex; gap: 16px; flex-wrap: wrap; color: var(--muted); font-size: 0.95rem; }}
    .overview {{ background: var(--panel); border: 1px solid var(--border); border-radius: 24px; box-shadow: var(--shadow); padding: 24px; margin-bottom: 28px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 12px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }}
    th {{ font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
    .cards {{ display: grid; gap: 20px; }}
    .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 24px; box-shadow: var(--shadow); padding: 24px; }}
    .card-header {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }}
    .pill {{ background: #e7f0ff; color: #1146a6; border-radius: 999px; padding: 8px 12px; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; }}
    .muted {{ color: var(--muted); }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 16px 0 20px; }}
    .stats div, .details-grid div {{ background: #f7f9fc; border: 1px solid var(--border); border-radius: 16px; padding: 12px 14px; }}
    .stats span, .details-grid span {{ display: block; color: var(--muted); font-size: 0.82rem; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em; }}
    .stats strong {{ font-size: 1.05rem; }}
    .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 18px; }}
    figure {{ margin: 0; }}
    figcaption {{ margin-bottom: 10px; font-weight: 700; }}
    .chart {{ width: 100%; height: auto; display: block; }}
    .chart rect {{ fill: #fbfdff; stroke: var(--border); }}
    .chart .baseline {{ stroke: #d4dae3; stroke-width: 1; }}
    .details-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    code {{ font-family: \"IBM Plex Mono\", \"SFMono-Regular\", monospace; font-size: 0.9rem; word-break: break-word; }}
    .links {{ margin-top: 16px; }}
    .footer-panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 24px; box-shadow: var(--shadow); padding: 24px; margin-top: 28px; }}
    ul {{ margin-bottom: 0; }}
    @media (max-width: 720px) {{
      main {{ width: min(100% - 20px, 1180px); padding-top: 20px; }}
      .hero, .overview, .card, .footer-panel {{ padding: 20px; border-radius: 20px; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class=\"hero\">
      <h1>RoboWBC Policy Showcase</h1>
      <p>This artifact is generated automatically in CI from fixture-backed, no-download policy runs. It is meant to give a quick visual and operational glance at what the repository can execute today: consistent control-loop metrics, per-policy target traces, and downloadable Rerun recordings from the same synthetic transport setup.</p>
      <div class=\"meta-row\">
        <span>Generated: {html.escape(generated_at)}</span>
        <span>Commit: {commit_html or 'local'}</span>
        <span>{run_html}</span>
      </div>
    </section>

    <section class=\"overview\">
      <h2>Compared policies</h2>
      <p class=\"muted\">All runs use checked-in mock robot configs plus bundled test ONNX fixtures. The goal is repeatable CI coverage and a stable visual status page, not vendor-checkpoint realism.</p>
      <table>
        <thead>
          <tr><th>Policy</th><th>Coverage</th><th>Ticks</th><th>Avg inference</th><th>Achieved rate</th><th>Dropped frames</th></tr>
        </thead>
        <tbody>
          {''.join(overview_rows)}
        </tbody>
      </table>
    </section>

    <section class=\"cards\">
      {''.join(cards)}
    </section>

    <section class=\"footer-panel\">
      <h2>Not yet in the fixture showcase</h2>
      <ul>{excluded}</ul>
    </section>
  </main>
</body>
</html>"""

    (output_dir / 'index.html').write_text(html_doc, encoding='utf-8')
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


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

    reports = [run_policy(repo_root, binary, output_dir, policy, env) for policy in POLICIES]
    render_html(reports, output_dir)
    print(f"wrote showcase report to {output_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
