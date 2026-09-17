"""Merge full-operator runs into the platform CSV and diagnostics report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable

try:
    from .classify_operator_semantics import classify as _classify_semantics
    from .classify_operator_semantics_9cat import category_code as _coarse_category_code
except ImportError:
    from classify_operator_semantics import classify as _classify_semantics
    from classify_operator_semantics_9cat import category_code as _coarse_category_code


PLATFORM_FIELDS = [
    "platform",
    "op_name",
    "dtype",
    "shape_config",
    "ic_latency_us",
    "ic_tflops",
    "pt_latency_us",
    "pt_tflops",
    "date",
    "tester",
    "remarks",
    "category_code",
    "infiniops_commit",
    "infinirt_commit",
]

CATEGORY_CODES = {
    "math",
    "logic",
    "reduction",
    "linalg",
    "neural",
    "indexing",
    "special",
    "dense",
    "sparse",
}

DIAGNOSTIC_FIELDS = [
    "op_name",
    "dtype",
    "benchmark_profile",
    "shape_config",
    "status",
    "implementation_index",
    "active_implementation_indices",
    "error",
    "worker_exit_code",
    "duration_s",
]

_RECIPE_RE = re.compile(r"\s*([^=;]+)=([^\[]+)\[([^\]]*)\]:([^;]+)")
_MATRIX_OPS = {
    "mm",
    "matmul",
    "linalg_matmul",
    "_int_mm",
    "addmm",
    "_addmm_activation",
    "bmm",
    "addbmm",
    "baddbmm",
    "linear",
}


def _load_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload.get("results", []))
    return rows


def parse_recipes(recipe_text: str) -> list[tuple[str, list[int]]]:
    recipes = []
    for match in _RECIPE_RE.finditer(recipe_text):
        raw_shape = match.group(3).strip()
        if not raw_shape:
            shape: list[int] = []
        elif "," in raw_shape:
            continue
        else:
            try:
                shape = [int(value) for value in raw_shape.split("x") if value]
            except ValueError:
                continue
        recipes.append((match.group(1).strip(), shape))
    return recipes


def _shape_text(shape: list[int]) -> str:
    return "[" + ", ".join(str(value) for value in shape) + "]"


def shape_config(row: dict[str, Any]) -> str:
    recipes = parse_recipes(str(row.get("input_recipe", "")))
    if not recipes:
        return str(row.get("input_recipe", ""))

    shapes = [shape for _, shape in recipes]
    op_name = str(row.get("op", ""))
    if op_name in _MATRIX_OPS and len(shapes) >= 2:
        a_shape, b_shape = shapes[-2:]
        return f"A={_shape_text(a_shape)}, B={_shape_text(b_shape)}"
    if len(shapes) == 1 or all(shape == shapes[0] for shape in shapes[1:]):
        return _shape_text(shapes[0])
    return ", ".join(f"{name}={_shape_text(shape)}" for name, shape in recipes)


def _flops(row: dict[str, Any]) -> float | None:
    op_name = str(row.get("op", ""))
    if op_name not in _MATRIX_OPS:
        return None
    shapes = [shape for _, shape in parse_recipes(str(row.get("input_recipe", "")))]
    if len(shapes) < 2:
        return None
    a_shape, b_shape = shapes[-2:]
    if not a_shape or not b_shape or len(a_shape) < 2 or len(b_shape) < 2:
        return None
    if op_name == "linear":
        if a_shape[-1] != b_shape[-1]:
            return None
        m = math.prod(a_shape[:-1])
        return float(2 * m * a_shape[-1] * b_shape[-2])
    if a_shape[-1] != b_shape[-2]:
        return None
    m = math.prod(a_shape[:-1])
    return float(2 * m * a_shape[-1] * b_shape[-1])


def _dtype(value: Any) -> str:
    return {
        "float16": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "float64": "fp64",
    }.get(str(value), str(value))


def _number(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.2f}"


def _tflops(flops: float | None, latency_us: Any) -> str:
    if flops is None or latency_us in (None, 0, ""):
        return "NA"
    return f"{flops / (float(latency_us) * 1e6):.4f}"


def _implementation_kind(row: dict[str, Any], platform: str) -> str:
    slot = row.get("implementation_index")
    return "torch-backend" if slot == 8 else f"{platform}-native"


def _category_code(row: dict[str, Any]) -> str:
    op_name = str(row.get("op", ""))
    semantic_category, _ = _classify_semantics(op_name)
    code = _coarse_category_code(
        {"op_name": op_name, "semantic_category": semantic_category}
    )
    return code if code in CATEGORY_CODES else "math"


def platform_row(
    row: dict[str, Any],
    *,
    platform: str,
    run_date: str,
    tester: str,
    infiniops_commit: str = "NA",
    infinirt_commit: str = "NA",
) -> dict[str, str]:
    flops = _flops(row)
    active = row.get("active_implementation_indices", [])
    remarks = "; ".join(
        [
            f"reference={row.get('ref_mode', '')}",
            f"implementation={_implementation_kind(row, platform)}",
            f"slot={row.get('implementation_index', '')}",
            f"active={json.dumps(active, separators=(',', ':'))}",
            f"validation={row.get('validation_mode', '')}",
        ]
    )
    return {
        "platform": platform,
        "op_name": str(row.get("op", "")),
        "dtype": _dtype(row.get("dtype", "")),
        "shape_config": shape_config(row),
        "ic_latency_us": _number(row.get("infini_us")),
        "ic_tflops": _tflops(flops, row.get("infini_us")),
        "pt_latency_us": _number(row.get("ref_us")),
        "pt_tflops": _tflops(flops, row.get("ref_us")),
        "date": run_date,
        "tester": tester,
        "remarks": remarks,
        "category_code": _category_code(row),
        "infiniops_commit": infiniops_commit,
        "infinirt_commit": infinirt_commit,
    }


def diagnostic_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "op_name": row.get("op", ""),
        "dtype": _dtype(row.get("dtype", "")),
        "benchmark_profile": row.get("benchmark_profile", ""),
        "shape_config": shape_config(row),
        "status": row.get("status", ""),
        "implementation_index": row.get("implementation_index", ""),
        "active_implementation_indices": json.dumps(
            row.get("active_implementation_indices", []), separators=(",", ":")
        ),
        "error": row.get("error", ""),
        "worker_exit_code": row.get("worker_exit_code", ""),
        "duration_s": row.get("duration_s", ""),
    }


def _dedupe(rows: Iterable[dict[str, Any]], *, successful: bool) -> list[dict[str, Any]]:
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if (row.get("status") == "success") != successful:
            continue
        key = (
            row.get("op"),
            row.get("dtype"),
            row.get("input_recipe"),
            row.get("implementation_index"),
        )
        selected.setdefault(key, row)
    return sorted(
        selected.values(),
        key=lambda row: (str(row.get("op", "")), str(row.get("dtype", "")), shape_config(row)),
    )


def write_report(
    path: Path,
    successes: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    source_count: int,
    platform: str,
) -> None:
    status_counts = Counter(str(row.get("status", "unknown")) for row in failures)
    implementation_counts = Counter(
        _implementation_kind(row, platform) for row in successes
    )
    ratios = [
        float(row["ref_us"]) / float(row["infini_us"])
        for row in successes
        if row.get("ref_us") and row.get("infini_us")
    ]
    lines = [
        f"# InfiniOps {platform.title()} benchmark report",
        "",
        f"- Source result sets: {source_count}",
        f"- Successful comparable cases: {len(successes)}",
        f"- Unique non-success cases: {len(failures)}",
        f"- Unique successful operators: {len({row.get('op') for row in successes})}",
        f"- Native implementation cases: {implementation_counts.get(f'{platform}-native', 0)}",
        f"- PyTorch backend wrapper cases: {implementation_counts.get('torch-backend', 0)}",
        f"- Median `PyTorch / InfiniOps` latency ratio: {statistics.median(ratios):.3f}x"
        if ratios
        else "- Median `PyTorch / InfiniOps` latency ratio: NA",
        "",
        "## Diagnostic status counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"| `{status}` | {count} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", type=Path, help="results.json files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--platform", default="nvidia")
    parser.add_argument("--date", dest="run_date", default=date.today().strftime("%Y/%m/%d"))
    parser.add_argument("--tester", default="baoming")
    parser.add_argument(
        "--infiniops-commit",
        default="NA",
        help="InfiniOps Git commit used for the benchmark",
    )
    parser.add_argument(
        "--infinirt-commit",
        default="NA",
        help="InfiniRT Git commit used for the benchmark",
    )
    args = parser.parse_args(argv)

    rows = _load_rows(args.sources)
    successes = _dedupe(rows, successful=True)
    failures = _dedupe(rows, successful=False)
    for path in (args.output, args.diagnostics, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLATFORM_FIELDS)
        writer.writeheader()
        for row in successes:
            writer.writerow(
                platform_row(
                    row,
                    platform=args.platform,
                    run_date=args.run_date,
                    tester=args.tester,
                    infiniops_commit=args.infiniops_commit,
                    infinirt_commit=args.infinirt_commit,
                )
            )

    with args.diagnostics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIAGNOSTIC_FIELDS)
        writer.writeheader()
        for row in failures:
            writer.writerow(diagnostic_row(row))

    write_report(args.report, successes, failures, len(args.sources), args.platform)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
