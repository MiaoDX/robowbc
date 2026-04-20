#!/usr/bin/env python3
"""Render a Markdown summary from normalized NVIDIA benchmark artifacts."""

from __future__ import annotations

import argparse
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


def consistent_field(artifacts: list[dict[str, Any]], field: str) -> str:
    values = {str(artifact.get(field, "")) for artifact in artifacts if artifact}
    if not values:
        return "n/a"
    if len(values) == 1:
        return values.pop()
    return "multiple"


def render_summary(registry: dict[str, Any], output_root: Path) -> str:
    official_artifacts = []
    robowbc_artifacts = []

    lines: list[str] = [
        "# NVIDIA Comparison Summary",
        "",
        "Generated from normalized artifacts under `artifacts/benchmarks/nvidia/`.",
        "",
    ]

    for case in registry["cases"]:
        case_id = case["case_id"]
        official = load_artifact(artifact_path(output_root, "official", case_id))
        robowbc = load_artifact(artifact_path(output_root, "robowbc", case_id))
        if official:
            official_artifacts.append(official)
        if robowbc:
            robowbc_artifacts.append(robowbc)

    lines.extend(
        [
            "## Provenance",
            "",
            f"- RoboWBC commit: `{consistent_field(robowbc_artifacts, 'robowbc_commit')}`",
            f"- Official upstream commit: `{consistent_field(official_artifacts, 'upstream_commit')}`",
            f"- Provider: `{consistent_field(official_artifacts + robowbc_artifacts, 'provider')}`",
            f"- Host fingerprint: `{consistent_field(official_artifacts + robowbc_artifacts, 'host_fingerprint')}`",
            "",
            "## Case Matrix",
            "",
            "| Case ID | RoboWBC | Official NVIDIA | RoboWBC / Official (p50) | Why it matters |",
            "|---------|---------|------------------|---------------------------|----------------|",
        ]
    )

    for case in registry["cases"]:
        case_id = case["case_id"]
        official = load_artifact(artifact_path(output_root, "official", case_id))
        robowbc = load_artifact(artifact_path(output_root, "robowbc", case_id))
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{case_id}`",
                    artifact_cell(robowbc),
                    artifact_cell(official),
                    ratio_string(robowbc, official),
                    str(case["interpretation"]),
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

    for case in registry["cases"]:
        case_id = case["case_id"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{case_id}`",
                    f"`robowbc/{case_id.replace('/', '__')}.json`",
                    f"`official/{case_id.replace('/', '__')}.json`",
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
            "scripts/bench_robowbc_compare.sh --all",
            "scripts/bench_nvidia_official.sh --all",
            "python3 scripts/render_nvidia_benchmark_summary.py --output artifacts/benchmarks/nvidia/SUMMARY.md",
            "```",
            "",
            "If a future environment is missing models or build prerequisites, the wrappers will emit",
            "blocked artifacts instead of silently substituting a different path.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("artifacts/benchmarks/nvidia/cases.json"),
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
    args = parser.parse_args()

    registry = load_registry(args.registry)
    rendered = render_summary(registry, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
