#!/usr/bin/env python3
"""Render Markdown and optional HTML summaries from normalized NVIDIA benchmark artifacts."""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
NORMALIZER_PATH = ROOT_DIR / "scripts/normalize_nvidia_benchmarks.py"


def load_normalizer() -> Any:
    spec = importlib.util.spec_from_file_location("normalize_nvidia_benchmarks", NORMALIZER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load normalizer module from {NORMALIZER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NORMALIZER = load_normalizer()
PROVIDER_ORDER = NORMALIZER.PROVIDER_ORDER
IMPLEMENTATION_ORDER = NORMALIZER.IMPLEMENTATION_ORDER
BASELINE_IMPLEMENTATION = "ort-cpp-sonic"
RUST_IMPLEMENTATION = "ort-rs"


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


def format_provider_family(provider: str) -> str:
    return NORMALIZER.provider_family(provider)


def format_implementation(implementation: str) -> str:
    return NORMALIZER.implementation_label(implementation)


def rendered_variant_labels() -> list[str]:
    variants = [NORMALIZER.provider_family("cpu")]
    for provider in PROVIDER_ORDER:
        if provider == "cpu":
            continue
        for implementation in IMPLEMENTATION_ORDER:
            variants.append(NORMALIZER.variant_label(provider, implementation))
    return variants


def ratio_string(
    ort_rs: dict[str, Any] | None, ort_cpp_sonic: dict[str, Any] | None
) -> str:
    if not ort_rs or not ort_cpp_sonic:
        return "n/a"
    if ort_rs.get("status") != "ok" or ort_cpp_sonic.get("status") != "ok":
        return "n/a"
    ort_rs_p50 = ort_rs.get("p50_ns")
    ort_cpp_sonic_p50 = ort_cpp_sonic.get("p50_ns")
    if (
        not isinstance(ort_rs_p50, int)
        or not isinstance(ort_cpp_sonic_p50, int)
        or ort_cpp_sonic_p50 == 0
    ):
        return "n/a"
    return f"{ort_rs_p50 / ort_cpp_sonic_p50:.2f}x"


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


def canonical_artifact_path(
    output_root: Path, implementation: str, provider: str, case_id: str
) -> Path:
    return (
        output_root
        / NORMALIZER.implementation_artifact_dir(implementation)
        / provider
        / f"{case_id.replace('/', '__')}.json"
    )


def legacy_artifact_path(
    output_root: Path, implementation: str, provider: str, case_id: str
) -> Path:
    return (
        output_root
        / NORMALIZER.legacy_artifact_dir(implementation)
        / provider
        / f"{case_id.replace('/', '__')}.json"
    )


def legacy_cpu_artifact_path(output_root: Path, implementation: str, case_id: str) -> Path:
    return (
        output_root
        / NORMALIZER.legacy_artifact_dir(implementation)
        / f"{case_id.replace('/', '__')}.json"
    )


def relpath(root: Path, path: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def locate_artifact(
    output_root: Path, implementation: str, provider: str, case_id: str
) -> tuple[dict[str, Any] | None, str]:
    canonical_path = canonical_artifact_path(output_root, implementation, provider, case_id)
    for candidate in (
        canonical_path,
        legacy_artifact_path(output_root, implementation, provider, case_id),
        legacy_cpu_artifact_path(output_root, implementation, case_id)
        if provider == "cpu"
        else None,
    ):
        if candidate is None:
            continue
        artifact = load_artifact(candidate)
        if artifact is not None:
            return artifact, relpath(output_root, candidate)
    return None, relpath(output_root, canonical_path)


def consistent_field(artifacts: list[dict[str, Any]], field: str) -> str:
    values = {
        str(value)
        for artifact in artifacts
        if artifact and (value := artifact.get(field)) not in (None, "")
    }
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


def build_provider_section(
    registry: dict[str, Any], output_root: Path, provider: str
) -> dict[str, Any]:
    artifacts_by_implementation: dict[str, list[dict[str, Any]]] = {
        implementation: [] for implementation in IMPLEMENTATION_ORDER
    }
    rows: list[dict[str, Any]] = []

    for case in registry["cases"]:
        case_id = case["case_id"]
        row_artifacts: dict[str, dict[str, Any] | None] = {}
        row_relpaths: dict[str, str] = {}
        for implementation in IMPLEMENTATION_ORDER:
            artifact, artifact_relpath = locate_artifact(output_root, implementation, provider, case_id)
            row_artifacts[implementation] = artifact
            row_relpaths[implementation] = artifact_relpath
            if artifact is not None:
                artifacts_by_implementation[implementation].append(artifact)

        rows.append(
            {
                "provider": format_provider_family(provider),
                "provider_id": provider,
                "provider_label": format_provider_family(provider),
                "case_id": case_id,
                "group": case_group(case_id),
                "description": str(case.get("description", "")),
                "interpretation": str(case["interpretation"]),
                "artifacts": row_artifacts,
                "artifact_relpaths": row_relpaths,
                "ratio": ratio_string(
                    row_artifacts[RUST_IMPLEMENTATION],
                    row_artifacts[BASELINE_IMPLEMENTATION],
                ),
            }
        )

    artifacts = [
        artifact
        for implementation in IMPLEMENTATION_ORDER
        for artifact in artifacts_by_implementation[implementation]
    ]
    return {
        "provider": format_provider_family(provider),
        "provider_id": provider,
        "provider_label": format_provider_family(provider),
        "rows": rows,
        "case_count": len(rows),
        "ok_pair_count": sum(
            1
            for row in rows
            if row["artifacts"][BASELINE_IMPLEMENTATION] is not None
            and row["artifacts"][RUST_IMPLEMENTATION] is not None
            and row["artifacts"][BASELINE_IMPLEMENTATION].get("status") == "ok"
            and row["artifacts"][RUST_IMPLEMENTATION].get("status") == "ok"
        ),
        "blocked_count": sum(
            1
            for row in rows
            if status_class(row["artifacts"][BASELINE_IMPLEMENTATION]) != "ok"
            or status_class(row["artifacts"][RUST_IMPLEMENTATION]) != "ok"
        ),
        "provenance": {
            "robowbc_commit": consistent_field(artifacts, "robowbc_commit"),
            "upstream_commit": consistent_field(artifacts, "upstream_commit"),
            "host_fingerprint": consistent_field(artifacts, "host_fingerprint"),
        },
    }


def build_summary(registry: dict[str, Any], output_root: Path) -> dict[str, Any]:
    provider_sections = [
        build_provider_section(registry, output_root, provider) for provider in PROVIDER_ORDER
    ]
    artifacts = [
        artifact
        for section in provider_sections
        for row in section["rows"]
        for artifact in row["artifacts"].values()
        if artifact is not None
    ]
    return {
        "providers": [format_provider_family(provider) for provider in PROVIDER_ORDER],
        "provider_ids": list(PROVIDER_ORDER),
        "implementations": [format_implementation(implementation) for implementation in IMPLEMENTATION_ORDER],
        "variants": rendered_variant_labels(),
        "provider_sections": provider_sections,
        "case_count": len(registry["cases"]),
        "row_count": sum(section["case_count"] for section in provider_sections),
        "ok_pair_count": sum(section["ok_pair_count"] for section in provider_sections),
        "blocked_count": sum(section["blocked_count"] for section in provider_sections),
        "provenance": {
            "robowbc_commit": consistent_field(artifacts, "robowbc_commit"),
            "upstream_commit": consistent_field(artifacts, "upstream_commit"),
            "host_fingerprint": consistent_field(artifacts, "host_fingerprint"),
        },
    }


def rerun_commands() -> str:
    return "\n".join(
        [
            'for provider in cpu cuda tensor_rt; do',
            '  python3 scripts/bench_robowbc_compare.py --all --provider "$provider"',
            '  python3 scripts/bench_nvidia_official.py --all --provider "$provider"',
            "done",
            "python3 scripts/render_nvidia_benchmark_summary.py --output artifacts/benchmarks/nvidia/SUMMARY.md",
            "# or build the full static site bundle:",
            "python3 scripts/build_site.py --output-dir /tmp/robowbc-site",
        ]
    )


def render_markdown(summary: dict[str, Any]) -> str:
    provenance = summary["provenance"]
    provider_list = ", ".join(f"`{provider}`" for provider in summary["providers"])
    variant_list = ", ".join(f"`{variant}`" for variant in summary["variants"])
    implementation_list = ", ".join(f"`{implementation}`" for implementation in summary["implementations"])
    lines: list[str] = [
        "# NVIDIA Comparison Summary",
        "",
        "Generated from normalized artifacts under `artifacts/benchmarks/nvidia/`",
        "using the tracked case registry in `benchmarks/nvidia/cases.json`.",
        "",
        "## Provenance",
        "",
        f"- Variant families rendered: {provider_list}",
        f"- Canonical variants: {variant_list}",
        f"- Implementations compared: {implementation_list}",
        f"- RoboWBC commit: `{provenance['robowbc_commit']}`",
        f"- Official upstream commit: `{provenance['upstream_commit']}`",
        f"- Host fingerprint: `{provenance['host_fingerprint']}`",
        f"- Canonical cases per provider family: `{summary['case_count']}`",
        f"- Total rendered rows: `{summary['row_count']}`",
        f"- Matched ok pairs: `{summary['ok_pair_count']}`",
        f"- Blocked or missing rows: `{summary['blocked_count']}`",
        "",
    ]

    for section in summary["provider_sections"]:
        section_provenance = section["provenance"]
        lines.extend(
            [
                f"## {section['provider_label']}",
                "",
                f"- Provider family id: `{section['provider']}`",
                f"- Benchmark provider request: `{section['provider_id']}`",
                f"- Matched ok pairs: `{section['ok_pair_count']}`",
                f"- Blocked or missing rows: `{section['blocked_count']}`",
                f"- RoboWBC commit: `{section_provenance['robowbc_commit']}`",
                f"- Official upstream commit: `{section_provenance['upstream_commit']}`",
                f"- Host fingerprint: `{section_provenance['host_fingerprint']}`",
                "",
                "| Path Group | Case ID | ORT-cpp-sonic | ORT-rs | ORT-rs / ORT-cpp-sonic (p50) | Why it matters |",
                "|------------|---------|----------------|--------|-------------------------------|----------------|",
            ]
        )

        for row in section["rows"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["group"],
                        f"`{row['case_id']}`",
                        artifact_cell(row["artifacts"][BASELINE_IMPLEMENTATION]),
                        artifact_cell(row["artifacts"][RUST_IMPLEMENTATION]),
                        row["ratio"],
                        row["interpretation"],
                    ]
                )
                + " |"
            )

        lines.extend(
            [
                "",
                "### Raw Artifacts",
                "",
                "| Case ID | ORT-cpp-sonic Artifact | ORT-rs Artifact |",
                "|---------|-------------------------|-----------------|",
            ]
        )

        for row in section["rows"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row['case_id']}`",
                        f"`{row['artifact_relpaths'][BASELINE_IMPLEMENTATION]}`",
                        f"`{row['artifact_relpaths'][RUST_IMPLEMENTATION]}`",
                    ]
                )
                + " |"
            )

        lines.append("")

    lines.extend(
        [
            "## Rerun",
            "",
            "```bash",
            rerun_commands(),
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
    provider_label_text = " / ".join(summary["providers"])
    variant_label_text = " / ".join(summary["variants"])

    def metric_card(label: str, value: str) -> str:
        return (
            f"<div class=\"metric-card\"><span>{html.escape(label)}</span>"
            f"<strong>{html.escape(value)}</strong></div>"
        )

    def render_section(section: dict[str, Any]) -> str:
        rows_html: list[str] = []
        for row in section["rows"]:
            ort_cpp_sonic = row["artifacts"][BASELINE_IMPLEMENTATION]
            ort_rs = row["artifacts"][RUST_IMPLEMENTATION]
            rows_html.append(
                f"""<tr>
  <td><span class="group-pill">{html.escape(row['group'])}</span></td>
  <td>
    <code>{html.escape(row['case_id'])}</code>
    <div class="case-detail">{html.escape(row['description'])}</div>
  </td>
  <td class="status-{status_class(ort_cpp_sonic)}">{html.escape(artifact_cell(ort_cpp_sonic))}</td>
  <td class="status-{status_class(ort_rs)}">{html.escape(artifact_cell(ort_rs))}</td>
  <td>{html.escape(row['ratio'])}</td>
  <td>{html.escape(row['interpretation'])}</td>
  <td>
    <a href="{html.escape(row['artifact_relpaths'][BASELINE_IMPLEMENTATION])}">ORT-cpp-sonic JSON</a><br />
    <a href="{html.escape(row['artifact_relpaths'][RUST_IMPLEMENTATION])}">ORT-rs JSON</a>
  </td>
</tr>"""
            )

        section_provenance = section["provenance"]
        return f"""<section class="panel provider-panel" id="provider-{html.escape(section['provider'])}">
      <div class="section-heading">
        <div>
          <span class="provider-pill">{html.escape(section['provider_label'])}</span>
          <h2>{html.escape(section['provider_label'])} Matrix</h2>
          <p class="muted">This family keeps the ORT-cpp-sonic baseline beside ORT-rs on the same requested provider. Decoupled WBC remains CPU-only in this phase, so non-CPU rows stay blocked instead of being relabeled.</p>
        </div>
      </div>
      <div class="section-metrics">
        {metric_card("Rows", str(section["case_count"]))}
        {metric_card("Matched ok pairs", str(section["ok_pair_count"]))}
        {metric_card("Blocked or missing rows", str(section["blocked_count"]))}
        {metric_card("Provider request", section["provider_id"])}
        {metric_card("RoboWBC commit", section_provenance["robowbc_commit"])}
        {metric_card("Official upstream commit", section_provenance["upstream_commit"])}
      </div>
      <p class="muted section-host">Host fingerprint: <code>{html.escape(section_provenance['host_fingerprint'])}</code></p>
      <table>
        <thead>
          <tr>
            <th>Path group</th>
            <th>Case</th>
            <th>ORT-cpp-sonic</th>
            <th>ORT-rs</th>
            <th>ORT-rs / ORT-cpp-sonic (p50)</th>
            <th>Why it matters</th>
            <th>Artifacts</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </section>"""

    provider_nav = "".join(
        f'<a class="hero-link" href="#provider-{html.escape(provider)}">{html.escape(provider)}</a>'
        for provider in summary["providers"]
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
    .metrics, .section-metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 20px; }}
    .metric-card {{ background: #f7f9fc; border: 1px solid var(--border); border-radius: 18px; padding: 14px 16px; }}
    .metric-card span {{ display: block; color: var(--muted); font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }}
    .metric-card strong {{ font-size: 1rem; }}
    .group-pill, .provider-pill {{ display: inline-flex; align-items: center; padding: 6px 10px; border-radius: 999px; border: 1px solid var(--border); background: #eef4ff; color: var(--accent); font-size: 0.82rem; font-weight: 600; }}
    .provider-panel h2 {{ margin-bottom: 10px; }}
    .section-heading {{ display: flex; flex-wrap: wrap; justify-content: space-between; gap: 16px; align-items: flex-start; }}
    .section-host {{ margin-top: 16px; margin-bottom: 20px; }}
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
      <p>This page is generated automatically from normalized ORT-cpp-sonic and ORT-rs benchmark artifacts. It keeps the benchmark vocabulary aligned with the codebase instead of mixing provider labels with legacy stack names.</p>
      <p class="muted">Canonical variant families: {html.escape(provider_label_text)}. Canonical rendered variants: {html.escape(variant_label_text)}. Decoupled WBC remains CPU-only in this phase and stays blocked on non-CPU rows rather than being approximated.</p>
      <div class="hero-links">
        <a class="hero-link" href="../../index.html">Site home</a>
        <a class="hero-link" href="SUMMARY.md">Markdown summary</a>
        <a class="hero-link" href="cases.json">Case registry</a>
        {provider_nav}
      </div>
      <div class="metrics">
        {metric_card("Canonical cases per family", str(summary["case_count"]))}
        {metric_card("Variant families", provider_label_text)}
        {metric_card("Canonical variants", variant_label_text)}
        {metric_card("Implementation columns", " / ".join(summary["implementations"]))}
        {metric_card("Matched ok pairs", str(summary["ok_pair_count"]))}
        {metric_card("Blocked or missing rows", str(summary["blocked_count"]))}
        {metric_card("RoboWBC commit", provenance["robowbc_commit"])}
        {metric_card("Official upstream commit", provenance["upstream_commit"])}
      </div>
      <p class="muted" style="margin-top: 18px;">Host fingerprint(s): <code>{html.escape(provenance['host_fingerprint'])}</code></p>
    </section>

    {''.join(render_section(section) for section in summary["provider_sections"])}

    <section class="panel">
      <h2>Rerun Commands</h2>
      <div class="callout">
        <pre><code>{html.escape(rerun_commands())}</code></pre>
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
