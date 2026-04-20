#!/usr/bin/env python3
"""Normalize NVIDIA comparison artifacts into one schema."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                _, value = line.split(":", 1)
                return value.strip()
    return platform.processor() or "unknown-cpu"


def default_host_fingerprint() -> str:
    return (
        f"{socket.gethostname()} | {platform.system()} {platform.release()} | "
        f"{platform.machine()} | {cpu_model()}"
    )


def percentile(values: list[float], pct: float) -> int | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0])
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower])
    fraction = position - lower
    interpolated = ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    return round(interpolated)


def load_registry(path: Path) -> dict[str, Any]:
    registry = load_json(path)
    if registry.get("schema_version") != 1:
        raise ValueError(f"unsupported registry schema version: {registry.get('schema_version')}")
    seen: set[str] = set()
    for case in registry.get("cases", []):
        case_id = case.get("case_id")
        if not case_id:
            raise ValueError("registry case is missing case_id")
        if case_id in seen:
            raise ValueError(f"duplicate case_id in registry: {case_id}")
        seen.add(case_id)
        for field in ("description", "command_fixture", "warmup_policy", "interpretation"):
            if field not in case:
                raise ValueError(f"registry case {case_id!r} is missing required field {field!r}")
    return registry


def registry_case(registry: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in registry["cases"]:
        if case["case_id"] == case_id:
            return case
    raise ValueError(f"unknown case_id: {case_id}")


def criterion_samples_ns(criterion_root: Path, criterion_id: str) -> tuple[list[float], str]:
    for benchmark_json in criterion_root.rglob("benchmark.json"):
        benchmark = load_json(benchmark_json)
        if benchmark.get("full_id") != criterion_id:
            continue
        sample_path = benchmark_json.parent / "sample.json"
        sample = load_json(sample_path)
        iters = sample.get("iters", [])
        times = sample.get("times", [])
        if len(iters) != len(times):
            raise ValueError(f"criterion sample mismatch for {criterion_id}")
        samples = [float(total_ns) / float(iter_count) for iter_count, total_ns in zip(iters, times)]
        return samples, str(sample_path)
    raise FileNotFoundError(
        f"could not find criterion benchmark {criterion_id!r} under {criterion_root}"
    )


def run_report_samples_ns(report_path: Path) -> tuple[list[float], float | None, str]:
    report = load_json(report_path)
    frames = report.get("frames", [])
    samples = [float(frame["inference_latency_ms"]) * 1_000_000.0 for frame in frames]
    hz = report.get("metrics", {}).get("achieved_frequency_hz")
    return samples, float(hz) if hz is not None else None, str(report_path)


def manual_samples_payload(input_path: Path) -> tuple[list[float], float | None, str]:
    payload = load_json(input_path)
    if isinstance(payload, list):
        return [float(value) for value in payload], None, str(input_path)
    samples = payload.get("samples_ns")
    if samples is None:
        raise ValueError(f"{input_path} must be a JSON list or an object with samples_ns")
    hz = payload.get("hz")
    return [float(value) for value in samples], float(hz) if hz is not None else None, str(input_path)


def build_artifact(
    *,
    case: dict[str, Any],
    stack: str,
    upstream_commit: str | None,
    robowbc_commit: str | None,
    provider: str,
    host_fingerprint: str | None,
    samples_ns: list[float],
    hz: float | None,
    notes: str,
    source_command: str | None,
    raw_source: str,
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "case_id": case["case_id"],
        "description": case["description"],
        "interpretation": case["interpretation"],
        "stack": stack,
        "upstream_commit": upstream_commit,
        "robowbc_commit": robowbc_commit,
        "provider": provider,
        "host_fingerprint": host_fingerprint or default_host_fingerprint(),
        "command_fixture": case["command_fixture"],
        "warmup_policy": case["warmup_policy"],
        "samples": len(samples_ns),
        "p50_ns": percentile(samples_ns, 0.50),
        "p95_ns": percentile(samples_ns, 0.95),
        "p99_ns": percentile(samples_ns, 0.99),
        "hz": round(hz, 6) if hz is not None else None,
        "notes": notes,
        "source_command": source_command,
        "raw_source": raw_source,
    }


def cmd_validate_registry(args: argparse.Namespace) -> int:
    load_registry(args.registry)
    return 0


def common_artifact_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--stack", required=True, choices=("robowbc", "official_nvidia"))
    parser.add_argument("--provider", required=True)
    parser.add_argument("--upstream-commit")
    parser.add_argument("--robowbc-commit")
    parser.add_argument("--host-fingerprint")
    parser.add_argument("--notes", default="")
    parser.add_argument("--source-command")
    parser.add_argument("--output", type=Path, required=True)


def cmd_normalize_criterion(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    case = registry_case(registry, args.case_id)
    criterion_id = case.get("criterion_id")
    if not criterion_id:
        raise ValueError(f"case {args.case_id!r} does not define criterion_id")
    samples_ns, raw_source = criterion_samples_ns(args.criterion_root, criterion_id)
    artifact = build_artifact(
        case=case,
        stack=args.stack,
        upstream_commit=args.upstream_commit,
        robowbc_commit=args.robowbc_commit,
        provider=args.provider,
        host_fingerprint=args.host_fingerprint,
        samples_ns=samples_ns,
        hz=None,
        notes=args.notes,
        source_command=args.source_command,
        raw_source=raw_source,
        status="ok",
    )
    dump_json(args.output, artifact)
    return 0


def cmd_normalize_run_report(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    case = registry_case(registry, args.case_id)
    samples_ns, hz, raw_source = run_report_samples_ns(args.input)
    artifact = build_artifact(
        case=case,
        stack=args.stack,
        upstream_commit=args.upstream_commit,
        robowbc_commit=args.robowbc_commit,
        provider=args.provider,
        host_fingerprint=args.host_fingerprint,
        samples_ns=samples_ns,
        hz=hz,
        notes=args.notes,
        source_command=args.source_command,
        raw_source=raw_source,
        status="ok",
    )
    dump_json(args.output, artifact)
    return 0


def cmd_normalize_samples(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    case = registry_case(registry, args.case_id)
    samples_ns, hz, raw_source = manual_samples_payload(args.input)
    artifact = build_artifact(
        case=case,
        stack=args.stack,
        upstream_commit=args.upstream_commit,
        robowbc_commit=args.robowbc_commit,
        provider=args.provider,
        host_fingerprint=args.host_fingerprint,
        samples_ns=samples_ns,
        hz=hz,
        notes=args.notes,
        source_command=args.source_command,
        raw_source=raw_source,
        status="ok",
    )
    dump_json(args.output, artifact)
    return 0


def cmd_emit_blocked(args: argparse.Namespace) -> int:
    registry = load_registry(args.registry)
    case = registry_case(registry, args.case_id)
    artifact = build_artifact(
        case=case,
        stack=args.stack,
        upstream_commit=args.upstream_commit,
        robowbc_commit=args.robowbc_commit,
        provider=args.provider,
        host_fingerprint=args.host_fingerprint,
        samples_ns=[],
        hz=None,
        notes=args.reason if not args.notes else f"{args.notes} | {args.reason}",
        source_command=args.source_command,
        raw_source=args.reason,
        status="blocked",
    )
    dump_json(args.output, artifact)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-registry")
    validate.add_argument("--registry", type=Path, required=True)
    validate.set_defaults(func=cmd_validate_registry)

    normalize_criterion = subparsers.add_parser("normalize-criterion")
    common_artifact_args(normalize_criterion)
    normalize_criterion.add_argument("--criterion-root", type=Path, required=True)
    normalize_criterion.set_defaults(func=cmd_normalize_criterion)

    normalize_run = subparsers.add_parser("normalize-run-report")
    common_artifact_args(normalize_run)
    normalize_run.add_argument("--input", type=Path, required=True)
    normalize_run.set_defaults(func=cmd_normalize_run_report)

    normalize_samples = subparsers.add_parser("normalize-samples")
    common_artifact_args(normalize_samples)
    normalize_samples.add_argument("--input", type=Path, required=True)
    normalize_samples.set_defaults(func=cmd_normalize_samples)

    blocked = subparsers.add_parser("emit-blocked")
    common_artifact_args(blocked)
    blocked.add_argument("--reason", required=True)
    blocked.set_defaults(func=cmd_emit_blocked)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
