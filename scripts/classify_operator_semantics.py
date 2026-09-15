from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


OUTPUT_FIELDS = [
    "op_name",
    "semantic_category",
    "semantic_subcategory",
    "variant",
    "status",
    "implementation",
    "benchmark_profile",
    "shape_config",
]


SPECIAL_EXACT = {
    "digamma", "erf", "erfc", "erfinv", "i0", "igamma", "igammac",
    "lgamma", "mvlgamma", "sinc", "xlogy",
}
LOSS_ROOTS = {
    "binary_cross_entropy", "huber_loss", "mse_loss", "multi_margin_loss",
    "multilabel_margin_loss", "nll_loss", "nll_loss2d", "smooth_l1_loss",
    "soft_margin_loss",
}
ACTIVATION_ROOTS = {
    "elu", "gelu", "glu", "hardshrink", "hardsigmoid", "hardswish",
    "hardtanh", "leaky_relu", "mish", "relu", "rrelu_with_noise", "silu",
    "sigmoid", "softplus", "softshrink", "tanh", "threshold",
    "internal_add_relu",
}
NORMALIZATION_ROOTS = {
    "batch_norm_elemt", "internal_batch_norm_with_update",
    "internal_native_batch_norm_legit", "internal_softmax",
    "internal_log_softmax", "log_softmax", "native_batch_norm", "softmax",
}
GEMM_ROOTS = {
    "addbmm", "addmm", "baddbmm", "bmm", "chain_matmul",
    "internal_addmm_activation", "internal_int_mm", "internal_scaled_mm",
    "linear", "linalg_matmul", "linalg_multi_dot", "matmul", "mm",
    "sparse_sampled_addmm", "sspaddmm",
}
CONTRACTION_ROOTS = {
    "addmv", "addr", "cross", "dot", "ger", "hspmm", "inner", "kron",
    "mv", "tensordot", "vdot",
}
DECOMPOSITION_ROOTS = {
    "cholesky", "geqrf", "linalg_cholesky", "linalg_cholesky_ex",
    "linalg_eig", "linalg_eigh", "linalg_eigvals", "linalg_eigvalsh",
    "linalg_householder_product", "linalg_ldl_factor",
    "linalg_ldl_factor_ex", "linalg_lu", "linalg_lu_factor",
    "linalg_lu_factor_ex", "linalg_qr", "linalg_svd", "linalg_svdvals",
    "lu_unpack", "orgqr", "ormqr", "qr", "svd",
}
SOLVE_ROOTS = {
    "cholesky_inverse", "cholesky_solve", "inverse", "linalg_inv",
    "linalg_inv_ex", "linalg_lstsq", "linalg_lu_solve", "linalg_pinv",
    "linalg_solve", "linalg_solve_ex", "linalg_solve_triangular",
    "linalg_tensorinv", "linalg_tensorsolve", "lu_solve", "triangular_solve",
}
MATRIX_STRUCTURE_ROOTS = {
    "diag", "linalg_matrix_power", "matrix_power", "tril", "triu",
}
MATRIX_NORM_ROOTS = {
    "frobenius_norm", "linalg_cond", "linalg_matrix_norm",
    "linalg_matrix_rank", "linalg_norm", "linalg_vector_norm",
    "linalg_vecdot", "nuclear_norm", "norm", "renorm",
}
MATRIX_INTERNAL_ROOTS = {
    "internal_linalg_det", "internal_linalg_eigh",
    "internal_linalg_slogdet", "internal_linalg_solve_ex",
    "internal_linalg_svd", "linalg_det", "linalg_slogdet", "slogdet",
}
REDUCTION_ROOTS = {
    "all", "amax", "amin", "aminmax", "any", "argmax", "argmin",
    "internal_logcumsumexp", "logsumexp", "max", "mean", "min", "nanmean",
    "nansum", "prod", "special_logsumexp", "std", "sum", "var",
}
CUMULATIVE_ROOTS = {"cummax", "cummin", "cumprod", "cumsum", "logcumsumexp"}
ORDER_SEARCH_ROOTS = {
    "argsort", "bucketize", "histc", "histogram", "isin", "kthvalue",
    "median", "mode", "msort", "nanmedian", "nanquantile", "quantile",
    "searchsorted", "sort", "topk",
}
INDEX_JOIN_ROOTS = {
    "cat", "column_stack", "concatenate", "dstack", "gather", "hstack",
    "index", "index_add", "index_copy", "index_fill", "index_reduce",
    "index_select", "internal_chunk_cat", "internal_stack", "masked_fill",
    "masked_select", "narrow_copy", "nonzero", "nonzero_static",
    "row_stack", "scatter", "scatter_add", "scatter_reduce", "stack",
    "take", "take_along_dim", "vstack",
}
ELEMENTWISE_ARITHMETIC_ROOTS = {
    "add", "addcdiv", "addcmul", "bitwise_and", "bitwise_left_shift",
    "bitwise_not", "bitwise_or", "bitwise_right_shift", "bitwise_xor",
    "copysign", "div", "divide", "eq", "fmax", "fmin", "fmod", "ge", "gt",
    "greater", "greater_equal", "heaviside", "hypot", "ldexp", "le",
    "lerp", "less", "less_equal", "logical_and", "logical_not",
    "logical_or", "logical_xor", "lt", "maximum", "minimum", "mul",
    "multiply", "ne", "nextafter", "not_equal", "pow", "remainder", "sub",
    "subtract", "true_divide", "where",
}
COMPLEX_ROOTS = {"angle", "complex", "conj_physical", "polar"}
GENERATION_ROOTS = {
    "bernoulli", "fill", "linspace", "logspace", "multinomial", "normal",
    "random", "set",
}


def strip_variant(op_name: str) -> tuple[str, str]:
    for suffix, variant in (
        ("_backward_data", "backward"),
        ("_backward", "backward"),
        ("_forward", "forward"),
    ):
        if op_name.endswith(suffix):
            return op_name[: -len(suffix)], variant
    return op_name, "forward"


def special_subcategory(root: str) -> str:
    if any(token in root for token in ("bessel", "airy", "spherical")):
        return "Bessel/球谐相关函数"
    if any(token in root for token in ("chebyshev", "hermite", "laguerre", "legendre")):
        return "正交多项式"
    if any(token in root for token in ("gamma", "psi", "zeta")):
        return "Gamma/Psi/Zeta"
    if any(token in root for token in ("erf", "ndtr", "ndtri")):
        return "误差/正态分布函数"
    if any(token in root for token in ("exp", "logit", "log1p", "entr", "sinc", "round")):
        return "指数/对数/数值函数"
    return "其他特殊函数"


def classify(op_name: str) -> tuple[str, str]:
    root, _ = strip_variant(op_name)

    if op_name.startswith("special_") or root in SPECIAL_EXACT:
        return "特殊数学函数", special_subcategory(root.removeprefix("special_"))
    if op_name.startswith("fft_") or op_name.startswith("internal_fft_"):
        return "谱变换/FFT", "FFT/IFFT"
    if root in LOSS_ROOTS:
        return "损失函数", "分类/回归/间隔损失"
    if root in ACTIVATION_ROOTS:
        return "神经网络激活", "常用激活"
    if root in NORMALIZATION_ROOTS:
        return "归一化", "softmax/BatchNorm/范数缩放"
    if "pool" in root or "unpool" in root:
        return "池化与上采样", "自适应池化" if root.startswith("adaptive") else "常规池化/反池化"
    if "upsample" in root:
        return "池化与上采样", "上采样/插值"
    if "conv" in root:
        return "卷积", "常规/深度/转置卷积"
    if root in {"im2col", "col2im"}:
        return "卷积", "卷积布局转换"
    if "pad" in root:
        return "填充", "reflection/replication padding"
    if root in {"internal_convert_indices_from_coo_to_csr", "internal_convert_indices_from_csr_to_coo"}:
        return "稀疏与格式转换", "COO/CSR转换"
    if root in {"hspmm", "sparse_sampled_addmm", "sspaddmm"}:
        return "矩阵与线性代数", "稀疏矩阵运算"
    if root in GEMM_ROOTS:
        return "矩阵与线性代数", "GEMM/线性层"
    if root in CONTRACTION_ROOTS:
        return "矩阵与线性代数", "点积/外积/张量收缩"
    if root in DECOMPOSITION_ROOTS:
        return "矩阵与线性代数", "分解/特征值"
    if root in SOLVE_ROOTS:
        return "矩阵与线性代数", "求逆/求解"
    if root in MATRIX_STRUCTURE_ROOTS:
        return "矩阵与线性代数", "矩阵结构/幂运算"
    if root in MATRIX_NORM_ROOTS:
        return "归约与统计", "范数/条件数/秩"
    if root in MATRIX_INTERNAL_ROOTS:
        return "矩阵与线性代数", "行列式/特征分解内部算子"
    if root.startswith("linalg_"):
        return "矩阵与线性代数", "其他linalg"
    if root in REDUCTION_ROOTS:
        return "归约与统计", "常用归约"
    if root in CUMULATIVE_ROOTS:
        return "归约与统计", "累积/前缀扫描"
    if root in ORDER_SEARCH_ROOTS:
        return "排序/搜索/分桶", "排序/搜索/分位数/直方图"
    if root in INDEX_JOIN_ROOTS:
        return "索引/拼接/重排", "索引/gather/scatter/拼接"
    if root in GENERATION_ROOTS:
        return "随机与生成", "随机采样/构造填充"
    if root in COMPLEX_ROOTS:
        return "复数与数值构造", "复数构造/共轭/辐角"
    if root in ELEMENTWISE_ARITHMETIC_ROOTS:
        return "元素级运算", "算术/比较/逻辑/条件选择"
    return "元素级数学变换", "一元数学/取整/钳制/符号/指数对数"


def implementation_from_main(row: dict[str, str]) -> str:
    remarks = row.get("remarks", "")
    if "implementation=metax-native" in remarks:
        return "metax-native"
    if "implementation=torch-backend" in remarks:
        return "torch-backend"
    return ""


def implementation_from_diag(row: dict[str, str]) -> str:
    slot = row.get("implementation_index", "")
    if slot == "8":
        return "torch-backend"
    if slot:
        return "metax-native"
    return ""


def load_rows(main_path: Path, diagnostics_path: Path) -> list[dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    with main_path.open(encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            name = source["op_name"]
            category, subcategory = classify(name)
            records[name] = {
                "op_name": name,
                "semantic_category": category,
                "semantic_subcategory": subcategory,
                "variant": strip_variant(name)[1],
                "status": "success",
                "implementation": implementation_from_main(source),
                "benchmark_profile": "decode-4k",
                "shape_config": source.get("shape_config", ""),
            }
    with diagnostics_path.open(encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            name = source["op_name"]
            if name in records:
                continue
            category, subcategory = classify(name)
            records[name] = {
                "op_name": name,
                "semantic_category": category,
                "semantic_subcategory": subcategory,
                "variant": strip_variant(name)[1],
                "status": source.get("status", ""),
                "implementation": implementation_from_diag(source),
                "benchmark_profile": source.get("benchmark_profile", ""),
                "shape_config": source.get("shape_config", ""),
            }
    return sorted(records.values(), key=lambda row: row["op_name"])


def write_summary(rows: list[dict[str, str]], path: Path) -> None:
    category_counts = Counter(row["semantic_category"] for row in rows)
    subcategory_counts: dict[str, Counter[str]] = defaultdict(Counter)
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        subcategory_counts[row["semantic_category"]][row["semantic_subcategory"]] += 1
        status_counts[row["semantic_category"]][row["status"]] += 1

    statuses = [
        "success", "reference_rejected", "backend_runtime_error", "timeout",
        "output_dtype_unsupported", "correctness_failed",
    ]
    lines = [
        "# InfiniOps 算子语义分类摘要", "",
        f"- 总算子数：{len(rows)}",
        f"- 一级语义类别：{len(category_counts)}", "",
        "## 一级类别", "",
        "| 类别 | 算子数 | 成功 | reference_rejected | backend_runtime_error | timeout | output_dtype_unsupported | correctness_failed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category, count in sorted(category_counts.items(), key=lambda item: (-item[1], item[0])):
        counters = status_counts[category]
        values = [str(counters.get(status, 0)) for status in statuses]
        lines.append(f"| {category} | {count} | " + " | ".join(values) + " |")
    lines.extend(["", "## 二级类别", "", "| 一级类别 | 二级类别 | 算子数 |", "|---|---|---:|"])
    for category in sorted(subcategory_counts, key=lambda item: (-sum(subcategory_counts[item].values()), item)):
        for subcategory, count in sorted(subcategory_counts[category].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| {category} | {subcategory} | {count} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.main, args.diagnostics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_summary(rows, args.summary)
    print(f"classified={len(rows)} output={args.output} summary={args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
