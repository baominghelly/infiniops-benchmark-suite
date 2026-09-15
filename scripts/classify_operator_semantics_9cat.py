from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


OUTPUT_FIELDS = [
    "op_name",
    "category_code",
    "category_zh",
    "semantic_subcategory",
    "variant",
    "status",
    "implementation",
    "benchmark_profile",
    "shape_config",
]

CATEGORIES = {
    "math": "基础数学类算子",
    "logic": "比较逻辑与位运算类算子",
    "reduction": "规约统计与排序类算子",
    "linalg": "矩阵计算与线性代数类算子",
    "neural": "神经网络常用算子",
    "indexing": "索引聚合与张量组织类算子",
    "special": "FFT与特殊函数类算子",
    "dense": "类CUDA稠密算子",
    "sparse": "类CUDA稀疏算子",
}

LOGIC_EXACT = {
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "eq",
    "ge",
    "gt",
    "greater",
    "greater_equal",
    "isin",
    "isneginf",
    "isposinf",
    "le",
    "less",
    "less_equal",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "lt",
    "ne",
    "not_equal",
    "signbit",
    "where",
}

SPARSE_EXACT = {
    "hspmm",
    "internal_convert_indices_from_coo_to_csr",
    "internal_convert_indices_from_csr_to_coo",
    "sparse_sampled_addmm",
    "sspaddmm",
}

TENSOR_CONSTRUCTION_EXACT = {
    "fill",
    "linspace",
    "logspace",
    "set",
}

SPARSE_FORMAT_EXACT = {
    "internal_convert_indices_from_coo_to_csr",
    "internal_convert_indices_from_csr_to_coo",
}

LINALG_REDUCTION_EXACT = {
    "frobenius_norm",
    "linalg_cond",
    "linalg_matrix_norm",
    "linalg_matrix_rank",
    "linalg_norm",
    "linalg_vector_norm",
    "linalg_vecdot",
    "nuclear_norm",
}


def strip_suffix(op_name: str) -> str:
    for suffix in ("_backward_data", "_backward", "_forward"):
        if op_name.endswith(suffix):
            return op_name[: -len(suffix)]
    return op_name


def category_code(row: dict[str, str]) -> str:
    op_name = row["op_name"]
    root = strip_suffix(op_name)
    old_category = row["semantic_category"]

    if root in SPARSE_EXACT or old_category == "稀疏与格式转换":
        return "sparse"
    if old_category in {"卷积", "池化与上采样", "填充"}:
        return "dense"
    if old_category in {"谱变换/FFT", "特殊数学函数"}:
        return "special"
    if old_category in {"神经网络激活", "归一化", "损失函数"}:
        return "neural"
    if old_category == "索引/拼接/重排":
        return "indexing"
    if old_category == "矩阵与线性代数":
        return "linalg"
    if old_category in {"归约与统计", "排序/搜索/分桶"}:
        if root in LINALG_REDUCTION_EXACT:
            return "linalg"
        return "reduction"
    if old_category == "随机与生成":
        return "indexing" if root in TENSOR_CONSTRUCTION_EXACT else "math"
    if root in LOGIC_EXACT:
        return "logic"
    return "math"


def load_rows(input_path: Path) -> list[dict[str, str]]:
    with input_path.open(encoding="utf-8", newline="") as handle:
        rows = []
        for source in csv.DictReader(handle):
            code = category_code(source)
            subcategory = source["semantic_subcategory"]
            if code == "sparse":
                root = strip_suffix(source["op_name"])
                subcategory = (
                    "稀疏格式转换"
                    if root in SPARSE_FORMAT_EXACT
                    else "稀疏矩阵运算"
                )
            rows.append(
                {
                    "op_name": source["op_name"],
                    "category_code": code,
                    "category_zh": CATEGORIES[code],
                    "semantic_subcategory": subcategory,
                    "variant": source["variant"],
                    "status": source["status"],
                    "implementation": source["implementation"],
                    "benchmark_profile": source["benchmark_profile"],
                    "shape_config": source["shape_config"],
                }
            )
    return sorted(rows, key=lambda row: row["op_name"])


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, str]], path: Path) -> None:
    status_by_category: dict[str, Counter[str]] = defaultdict(Counter)
    subcategory_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        status_by_category[row["category_code"]][row["status"]] += 1
        subcategory_counts[row["category_code"]][row["semantic_subcategory"]] += 1

    statuses = [
        "success",
        "reference_rejected",
        "backend_runtime_error",
        "timeout",
        "output_dtype_unsupported",
        "correctness_failed",
    ]
    lines = [
        "# InfiniOps 九类语义分类摘要",
        "",
        f"- 总算子数：{len(rows)}",
        f"- 一级分类数：{len(CATEGORIES)}",
        "",
        "## 一级分类",
        "",
        "| 英文码 | 中文类别 | 算子数 | 成功 | reference_rejected | backend_runtime_error | timeout | output_dtype_unsupported | correctness_failed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for code, chinese in CATEGORIES.items():
        counter = status_by_category[code]
        total = sum(counter.values())
        values = [str(counter.get(status, 0)) for status in statuses]
        lines.append(f"| `{code}` | {chinese} | {total} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## 二级语义",
            "",
            "| 英文码 | 中文类别 | 二级语义 | 算子数 |",
            "|---|---|---|---:|",
        ]
    )
    for code in CATEGORIES:
        for subcategory, count in sorted(
            subcategory_counts[code].items(), key=lambda item: (-item[1], item[0])
        ):
            lines.append(f"| `{code}` | {CATEGORIES[code]} | {subcategory} | {count} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    rows = load_rows(args.input)
    if set(row["category_code"] for row in rows) - set(CATEGORIES):
        raise ValueError("unknown category code")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output)
    write_summary(rows, args.summary)
    print(f"classified={len(rows)} output={args.output} summary={args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
