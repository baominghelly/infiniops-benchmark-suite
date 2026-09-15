"""Merge ordered full-operator benchmark attempts without mixing dtypes."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def load_results(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["results"] if isinstance(payload, dict) else payload


def median(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def parse_overlay(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("overlay must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    return name, Path(raw_path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    preferred = [
        "op",
        "aten_name",
        "status",
        "dtype",
        "source_run",
        "input_recipe",
        "infini_us",
        "ref_us",
        "speedup",
        "ref_mode",
        "validation_mode",
        "error",
    ]
    keys = {key for row in rows for key in row}
    fields = preferred + sorted(keys - set(preferred))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
            writer.writerow(encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--overlay", action="append", type=parse_overlay, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base_rows = load_results(args.base)
    if len(base_rows) != 492:
        raise RuntimeError(f"expected 492 base rows, got {len(base_rows)}")

    target_ops = {
        row["op"] for row in base_rows if row["status"] == "reference_rejected"
    }
    if len(target_ops) != 157:
        raise RuntimeError(
            f"expected 157 original reference_rejected rows, got {len(target_ops)}"
        )

    attempts: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}

    def add_run(name: str, rows: list[dict[str, Any]]) -> None:
        for original in rows:
            row = dict(original)
            row["source_run"] = name
            attempts.append(row)
            selected[row["op"]] = row

    add_run("full-v2", base_rows)
    run_metadata = [{"name": "full-v2", "path": str(args.base)}]
    for name, path in args.overlay:
        rows = load_results(path)
        add_run(name, rows)
        run_metadata.append({"name": name, "path": str(path), "rows": len(rows)})

    final_rows = [selected[name] for name in sorted(selected)]
    if len(final_rows) != 492:
        raise RuntimeError(f"expected 492 merged rows, got {len(final_rows)}")

    success_rows = [row for row in final_rows if row["status"] == "success"]
    target_rows = [row for row in final_rows if row["op"] in target_ops]
    target_success = [row for row in target_rows if row["status"] == "success"]
    remaining = [row for row in target_rows if row["status"] != "success"]

    dtype_summary = {}
    for dtype in sorted({row["dtype"] for row in success_rows}):
        dtype_rows = [row for row in success_rows if row["dtype"] == dtype]
        dtype_summary[dtype] = {
            "success": len(dtype_rows),
            "median_infini_us": median(
                [float(row["infini_us"]) for row in dtype_rows if row["infini_us"] is not None]
            ),
        }

    summary = {
        "attempted": len(final_rows),
        "success": len(success_rows),
        "coverage_pct": round(len(success_rows) / len(final_rows) * 100, 2),
        "by_status": dict(Counter(row["status"] for row in final_rows)),
        "original_reference_rejected": len(target_rows),
        "resolved_original_reference_rejected": len(target_success),
        "remaining_original_reference_rejected": len(remaining),
        "success_by_dtype": dtype_summary,
    }

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "merge_rule": "later ordered attempt replaces earlier result",
                    "runs": run_metadata,
                    "dtype_policy": "FP16 and FP32 timings are never aggregated together",
                },
                "summary": summary,
                "results": final_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "attempts.json").write_text(
        json.dumps(attempts, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output / "results.csv", final_rows)
    write_csv(output / "remaining.csv", remaining)

    lines = [
        "# A100 全算子覆盖合并结果",
        "",
        f"- 最终覆盖：{summary['success']}/492（{summary['coverage_pct']}%）",
        (
            "- 原 157 条 `reference_rejected`："
            f"已跑通 {summary['resolved_original_reference_rejected']}，"
            f"剩余 {summary['remaining_original_reference_rejected']}"
        ),
        "- 计时口径：FP16 与 FP32 分开统计，不计算跨 dtype 中位数。",
        "",
        "## 按 dtype 成功结果",
        "",
        "| dtype | 成功数 | InfiniOps 中位数 (us) |",
        "|---|---:|---:|",
    ]
    for dtype, values in dtype_summary.items():
        lines.append(
            f"| {dtype} | {values['success']} | {values['median_infini_us']} |"
        )

    lines.extend(
        [
            "",
            "## 最终状态",
            "",
            "| 状态 | 数量 |",
            "|---|---:|",
        ]
    )
    for status, count in sorted(summary["by_status"].items()):
        lines.append(f"| {status} | {count} |")

    lines.extend(
        [
            "",
            "## 原 157 条中仍未成功",
            "",
            "| 算子 | dtype | 状态 | 来源 | 错误 |",
            "|---|---|---|---|---|",
        ]
    )
    for row in remaining:
        error = str(row.get("error", "")).replace("|", "\\|").replace("\n", " ")
        if len(error) > 220:
            error = error[:217] + "..."
        lines.append(
            f"| {row['op']} | {row['dtype']} | {row['status']} | "
            f"{row['source_run']} | {error} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
