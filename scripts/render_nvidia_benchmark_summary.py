#!/usr/bin/env python3
"""Render Markdown and optional HTML summaries from normalized NVIDIA benchmark artifacts."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_registry(path: Path) -> dict[str, Any]:
    registry = load_json(path)
    if registry.get("schema_version") != 1:
        raise ValueError(f"unsupported registry schema version: {registry.get('schema_version')}")
    return registry


def load_artifact(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return load_json(path)


def format_ns(value: int | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f} ms"
    if value >= 1_000:
        return f"{value / 1_000:.3f} us"
    return f"{value} ns"


def format_hz(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f} Hz"


def ratio_string(robowbc: dict[str, Any] | None, official: dict[str, Any] | None) -> str:
    if not robowbc or not official:
        return "n/a"
    if robowbc.get("status") != "ok" or official.get("status") != "ok":
        return "n/a"
    robowbc_p50 = robowbc.get("p50_ns")
    official_p50 = official.get("p50_ns")
    if not isinstance(robowbc_p50, int) or not isinstance(official_p50, int) or official_p50 == 0:
        return "n/a"
    return f"{robowbc_p50 / official_p50:.2f}x"


def artifact_cell(artifact: dict[str, Any] | None) -> str:
    if artifact is None:
        return "missing"
    status = artifact.get("status", "unknown")
    if status != "ok":
        note = str(artifact.get("notes", "blocked"))
        return f"{status}; {note}"
    return (
        f"p50 {format_ns(artifact.get('p50_ns'))}; "
        f"p95 {format_ns(artifact.get('p95_ns'))}; "
        f"hz {format_hz(artifact.get('hz'))}"
    )


def artifact_path(output_root: Path, stack: str, case_id: str) -> Path:
    return output_root / stack / f"{case_id.replace('/', '__')}.json"


def artifact_relpath(stack: str, case_id: str) -> str:
    return f"{stack}/{case_id.replace('/', '__')}.json"


def consistent_field(artifacts: list[dict[str, Any]], field: str) -> str:
    values = {str(artifact.get(field, "")) for artifact in artifacts if artifact}
    if not values:
        return "n/a"
    if len(values) == 1:
        return values.pop()
    return "multiple"


def status_class(artifact: dict[str, Any] | None) -> str:
    if artifact is None:
        return "missing"
    if artifact.get("status") == "ok":
        return "ok"
    return "blocked"


def case_group(case_id: str) -> str:
    if case_id.startswith("gear_sonic/planner_only_"):
        return "Planner only"
    if case_id == "gear_sonic/encoder_decoder_only_tracking_tick":
        return "Encoder + decoder only"
    if case_id.startswith("gear_sonic/full_velocity_tick_"):
        return "Full velocity tick"
    if case_id == "gear_sonic/end_to_end_cli_loop":
        return "GEAR-Sonic end-to-end"
    if case_id.startswith("decoupled_wbc/"):
        return "Decoupled WBC"
    return "Other"


def build_summary(registry: dict[str, Any], output_root: Path) -> dict[str, Any]:
    official_artifacts = []
    robowbc_artifacts = []
    rows: list[dict[str, Any]] = []

    for case in registry["cases"]:
        case_id = case["case_id"]
        official = load_artifact(artifact_path(output_root, "official", case_id))
        robowbc = load_artifact(artifact_path(output_root, "robowbc", case_id))
        if official:
            official_artifacts.append(official)
        if robowbc:
            robowbc_artifacts.append(robowbc)

        rows.append(
            {
                "case_id": case_id,
                "group": case_group(case_id),
                "description": str(case.get("description", "")),
                "interpretation": str(case["interpretation"]),
                "official": official,
                "robowbc": robowbc,
                "ratio": ratio_string(robowbc, official),
                "official_relpath": artifact_relpath("official", case_id),
                "robowbc_relpath": artifact_relpath("robowbc", case_id),
            }
        )

    return {
        "provenance": {
            "robowbc_commit": consistent_field(robowbc_artifacts, "robowbc_commit"),
            "upstream_commit": consistent_field(official_artifacts, "upstream_commit"),
            "provider": consistent_field(official_artifacts + robowbc_artifacts, "provider"),
            "host_fingerprint": consistent_field(
                official_artifacts + robowbc_artifacts, "host_fingerprint"
            ),
        },
        "rows": rows,
        "case_count": len(rows),
        "ok_pair_count": sum(
            1
            for row in rows
            if row["official"] is not None
            and row["robowbc"] is not None
            and row["official"].get("status") == "ok"
            and row["robowbc"].get("status") == "ok"
        ),
        "blocked_count": sum(
            1
            for row in rows
            if status_class(row["official"]) != "ok" or status_class(row["robowbc"]) != "ok"
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    provenance = summary["provenance"]
    lines: list[str] = [
        "# NVIDIA Comparison Summary",
        "",
        "Generated from normalized artifacts under `artifacts/benchmarks/nvidia/`",
        "using the tracked case registry in `benchmarks/nvidia/cases.json`.",
        "",
        "## Provenance",
        "",
        f"- RoboWBC commit: `{provenance['robowbc_commit']}`",
        f"- Official upstream commit: `{provenance['upstream_commit']}`",
        f"- Provider: `{provenance['provider']}`",
        f"- Host fingerprint: `{provenance['host_fingerprint']}`",
        "",
        "## Case Matrix",
        "",
        "| Path Group | Case ID | RoboWBC | Official NVIDIA | RoboWBC / Official (p50) | Why it matters |",
        "|------------|---------|---------|------------------|---------------------------|----------------|",
    ]

    for row in summary["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["group"],
                    f"`{row['case_id']}`",
                    artifact_cell(row["robowbc"]),
                    artifact_cell(row["official"]),
                    row["ratio"],
                    row["interpretation"],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Raw Artifacts",
            "",
            "Each row above is backed by the paired normalized JSON artifacts below.",
            "",
            "| Case ID | RoboWBC Artifact | Official Artifact |",
            "|---------|------------------|-------------------|",
        ]
    )

    for row in summary["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['case_id']}`",
                    f"`{row['robowbc_relpath']}`",
                    f"`{row['official_relpath']}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Rerun",
            "",
            "```bash",
            "python3 scripts/bench_robowbc_compare.py --all",
            "python3 scripts/bench_nvidia_official.py --all",
            "python3 scripts/render_nvidia_benchmark_summary.py --output artifacts/benchmarks/nvidia/SUMMARY.md",
            "# or build the full static site bundle:",
            "python3 scripts/build_site.py --output-dir /tmp/robowbc-site",
            "```",
            "",
            "If a future environment is missing models or build prerequisites, the wrappers will emit",
            "blocked artifacts instead of silently substituting a different path.",
            "",
        ]
    )

    return "\n".join(lines)


def render_html(summary: dict[str, Any]) -> str:
    provenance = summary["provenance"]

    def metric_card(label: str, value: str) -> str:
        return (
            f"<div class=\"metric-card\"><span>{html.escape(label)}</span>"
            f"<strong>{html.escape(value)}</strong></div>"
        )

    rows_html: list[str] = []
    for row in summary["rows"]:
        rows_html.append(
            f"""<tr>
  <td><span class="group-pill">{html.escape(row['group'])}</span></td>
  <td>
    <code>{html.escape(row['case_id'])}</code>
    <div class="case-detail">{html.escape(row['description'])}</div>
  </td>
  <td class="status-{status_class(row['robowbc'])}">{html.escape(artifact_cell(row['robowbc']))}</td>
  <td class="status-{status_class(row['official'])}">{html.escape(artifact_cell(row['official']))}</td>
  <td>{html.escape(row['ratio'])}</td>
  <td>{html.escape(row['interpretation'])}</td>
  <td>
    <a href="{html.escape(row['robowbc_relpath'])}">RoboWBC JSON</a><br />
    <a href="{html.escape(row['official_relpath'])}">Official JSON</a>
  </td>
</tr>"""
        )

    rerun_cmds = "\n".join(
        [
            "python3 scripts/bench_robowbc_compare.py --all",
            "python3 scripts/bench_nvidia_official.py --all",
            "python3 scripts/render_nvidia_benchmark_summary.py --output artifacts/benchmarks/nvidia/SUMMARY.md --html-output /tmp/robowbc-site/benchmarks/nvidia/index.html",
            "# or build the full static site bundle:",
            "python3 scripts/build_site.py --output-dir /tmp/robowbc-site",
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RoboWBC NVIDIA Comparison</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #142033;
      --muted: #5f6f85;
      --border: #d9e0ea;
      --shadow: 0 18px 50px rgba(20, 32, 51, 0.08);
      --ok-bg: #e7f7ef;
      --ok-fg: #11643a;
      --blocked-bg: #fff1f2;
      --blocked-fg: #b42318;
      --missing-bg: #f1f5f9;
      --missing-fg: #475569;
      --accent: #0f5bd3;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; background: radial-gradient(circle at top, #eef7ff, var(--bg) 45%); color: var(--text); }}
    main {{ width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 40px 0 64px; }}
    h1, h2, p {{ margin-top: 0; }}
    a {{ color: var(--accent); }}
    code {{ font-family: "IBM Plex Mono", "SFMono-Regular", monospace; font-size: 0.92rem; word-break: break-word; }}
    .hero, .panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 28px; box-shadow: var(--shadow); padding: 28px; margin-bottom: 24px; }}
    .hero {{ background: linear-gradient(135deg, #ffffff, #ecf4ff); }}
    .hero p {{ max-width: 82ch; line-height: 1.6; }}
    .hero-links {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 18px; }}
    .hero-link {{ display: inline-flex; align-items: center; gap: 8px; padding: 10px 14px; border-radius: 999px; border: 1px solid var(--border); background: rgba(255, 255, 255, 0.85); text-decoration: none; font-weight: 600; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 20px; }}
    .metric-card {{ background: #f7f9fc; border: 1px solid var(--border); border-radius: 18px; padding: 14px 16px; }}
    .metric-card span {{ display: block; color: var(--muted); font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }}
    .metric-card strong {{ font-size: 1rem; }}
    .group-pill {{ display: inline-flex; align-items: center; padding: 6px 10px; border-radius: 999px; border: 1px solid var(--border); background: #eef4ff; color: var(--accent); font-size: 0.82rem; font-weight: 600; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 14px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }}
    th {{ font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
    .case-detail {{ margin-top: 8px; color: var(--muted); font-size: 0.94rem; line-height: 1.5; }}
    .status-ok {{ background: var(--ok-bg); color: var(--ok-fg); }}
    .status-blocked {{ background: var(--blocked-bg); color: var(--blocked-fg); }}
    .status-missing {{ background: var(--missing-bg); color: var(--missing-fg); }}
    .callout {{ background: #f7f9fc; border: 1px solid var(--border); border-radius: 18px; padding: 18px; }}
    .muted {{ color: var(--muted); }}
    pre {{ margin: 0; padding: 16px; border-radius: 18px; background: #0f172a; color: #e2e8f0; overflow-x: auto; }}
    @media (max-width: 720px) {{
      main {{ width: min(100% - 20px, 1180px); padding-top: 20px; }}
      .hero, .panel {{ padding: 20px; border-radius: 22px; }}
      th, td {{ padding: 12px 10px; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>RoboWBC NVIDIA Comparison</h1>
      <p>This page is generated automatically in CI from the normalized RoboWBC-vs-official benchmark artifacts. It keeps the matched-path comparison package visible on GitHub Pages without requiring readers to reconstruct the Markdown summary locally.</p>
      <p class="muted">Rows stay honest: if a future rerun loses models, prerequisites, or an executable upstream seam, the page surfaces the blocked artifact instead of approximating a nearby path.</p>
      <div class="hero-links">
        <a class="hero-link" href="../../index.html">Site home</a>
        <a class="hero-link" href="SUMMARY.md">Markdown summary</a>
        <a class="hero-link" href="cases.json">Case registry</a>
      </div>
      <div class="metrics">
        {metric_card("Cases", str(summary["case_count"]))}
        {metric_card("Matched ok pairs", str(summary["ok_pair_count"]))}
        {metric_card("Blocked or missing rows", str(summary["blocked_count"]))}
        {metric_card("Provider", provenance["provider"])}
        {metric_card("RoboWBC commit", provenance["robowbc_commit"])}
        {metric_card("Official upstream commit", provenance["upstream_commit"])}
      </div>
      <p class="muted" style="margin-top: 18px;">Host fingerprint: <code>{html.escape(provenance['host_fingerprint'])}</code></p>
    </section>

    <section class="panel">
      <h2>Case Matrix</h2>
      <table>
        <thead>
          <tr>
            <th>Path group</th>
            <th>Case</th>
            <th>RoboWBC</th>
            <th>Official NVIDIA</th>
            <th>p50 ratio</th>
            <th>Why it matters</th>
            <th>Artifacts</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </section>

    <section class="panel">
      <h2>Rerun Commands</h2>
      <div class="callout">
        <pre><code>{html.escape(rerun_cmds)}</code></pre>
      </div>
    </section>
  </main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("benchmarks/nvidia/cases.json"),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/benchmarks/nvidia"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmarks/nvidia/SUMMARY.md"),
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        help="Optional path for a static HTML report generated from the same artifacts.",
    )
    args = parser.parse_args()

    registry = load_registry(args.registry)
    summary = build_summary(registry, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summary), encoding="utf-8")
    if args.html_output is not None:
        args.html_output.parent.mkdir(parents=True, exist_ok=True)
        args.html_output.write_text(render_html(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
