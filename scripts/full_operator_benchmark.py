"""Schema-driven fixed-input performance benchmark for InfiniOps wrappers."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_DEVICE_PLUGIN_MODULES = {
    "mlu": "torch_mlu",
    "musa": "torch_musa",
    "npu": "torch_npu",
}


def _device_type(device: str) -> str:
    return str(device).split(":", 1)[0].lower()


def _load_device_plugin(device: str) -> None:
    device_type = _device_type(device)
    module_name = _DEVICE_PLUGIN_MODULES.get(device_type)
    if module_name is None:
        return
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"PyTorch device plugin '{module_name}' is required for device "
            f"'{device_type}'"
        ) from exc


def _accelerator_module(torch: Any, device: str) -> Any | None:
    return getattr(torch, _device_type(device), None)


def _manual_seed_all(torch: Any, device: str, seed: int) -> None:
    accelerator = _accelerator_module(torch, device)
    manual_seed_all = getattr(accelerator, "manual_seed_all", None)
    if manual_seed_all is not None:
        manual_seed_all(seed)


def _synchronize(torch: Any, device: str) -> None:
    accelerator = _accelerator_module(torch, device)
    synchronize = getattr(accelerator, "synchronize", None)
    if synchronize is not None:
        synchronize()


RESULT_FIELDS = [
    "op",
    "aten_name",
    "overload_name",
    "status",
    "implementation_index",
    "active_implementation_indices",
    "benchmark_profile",
    "dtype",
    "input_recipe",
    "infini_us",
    "ref_us",
    "speedup",
    "ref_mode",
    "validation_mode",
    "error",
    "worker_exit_code",
    "duration_s",
]

_BENCHMARK_PROFILE = os.environ.get("INFINIOPS_BENCHMARK_PROFILE", "compact")
_PROFILE_DEFAULT_SHAPES = {
    "compact": [8, 128],
    "decode-4k-small": [1, 128],
    "decode-4k": [1, 4096],
    "decode-8k": [1, 8192],
    "prefill-4k": [8, 4096],
}
if _BENCHMARK_PROFILE not in _PROFILE_DEFAULT_SHAPES:
    raise RuntimeError(
        "INFINIOPS_BENCHMARK_PROFILE must be one of "
        + ", ".join(sorted(_PROFILE_DEFAULT_SHAPES))
    )
_DEFAULT_SHAPE = _PROFILE_DEFAULT_SHAPES[_BENCHMARK_PROFILE]

_MATRIX_SHAPES = {
    "mm": [[64, 128], [128, 64]],
    "matmul": [[64, 128], [128, 64]],
    "bmm": [[4, 64, 128], [4, 128, 64]],
    "addmm": [[64, 64], [64, 128], [128, 64]],
    "addbmm": [[64, 64], [4, 64, 128], [4, 128, 64]],
    "baddbmm": [[4, 64, 64], [4, 64, 128], [4, 128, 64]],
    "addmv": [[64], [64, 128], [128]],
    "mv": [[64, 128], [128]],
    "dot": [[4096], [4096]],
    "vdot": [[4096], [4096]],
    "inner": [[64, 128], [64, 128]],
    "outer": [[128], [256]],
    "ger": [[128], [256]],
    "addr": [[128, 256], [128], [256]],
    "triangular_solve": [[32, 8], [32, 32]],
    "linalg_solve": [[32, 32], [32, 8]],
    "linalg_solve_ex": [[32, 32], [32, 8]],
    "linalg_solve_triangular": [[32, 32], [32, 8]],
    "_addmm_activation": [[64, 64], [64, 128], [128, 64]],
    "linalg_matmul": [[64, 128], [128, 64]],
    "_int_mm": [[32, 64], [64, 32]],
    "_linalg_solve_ex": [[32, 32], [32, 8]],
    "cholesky_solve": [[32, 8], [32, 32]],
    "linalg_tensorsolve": [[4, 4, 4, 4], [4, 4]],
    "linalg_lu_solve": [[32, 32], [32], [32, 8]],
    "lu_solve": [[32, 8], [32, 32], [32]],
    "linalg_householder_product": [[64, 32], [32]],
    "orgqr": [[64, 32], [32]],
    "ormqr": [[64, 32], [32], [16, 64]],
}

_PROFILE_MATRIX_SHAPES = {
    "decode-4k-small": {
        "mm": [[1, 128], [128, 256]],
        "matmul": [[1, 128], [128, 256]],
        "linalg_matmul": [[1, 128], [128, 256]],
        "addmm": [[1, 256], [1, 128], [128, 256]],
        "_addmm_activation": [[1, 256], [1, 128], [128, 256]],
        "linear": [[1, 128], [256, 128]],
    },
    "decode-4k": {
        "mm": [[1, 4096], [4096, 11008]],
        "matmul": [[1, 4096], [4096, 11008]],
        "linalg_matmul": [[1, 4096], [4096, 11008]],
        "addmm": [[1, 11008], [1, 4096], [4096, 11008]],
        "_addmm_activation": [[1, 11008], [1, 4096], [4096, 11008]],
        "linear": [[1, 4096], [11008, 4096]],
    },
    "decode-8k": {
        "mm": [[1, 8192], [8192, 28672]],
        "matmul": [[1, 8192], [8192, 28672]],
        "linalg_matmul": [[1, 8192], [8192, 28672]],
        "addmm": [[1, 28672], [1, 8192], [8192, 28672]],
        "_addmm_activation": [[1, 28672], [1, 8192], [8192, 28672]],
        "linear": [[1, 8192], [28672, 8192]],
    },
    "prefill-4k": {
        "mm": [[8, 4096], [4096, 11008]],
        "matmul": [[8, 4096], [4096, 11008]],
        "linalg_matmul": [[8, 4096], [4096, 11008]],
        "addmm": [[8, 11008], [8, 4096], [4096, 11008]],
        "_addmm_activation": [[8, 11008], [8, 4096], [4096, 11008]],
        "linear": [[8, 4096], [11008, 4096]],
    },
}
_MATRIX_SHAPES.update(_PROFILE_MATRIX_SHAPES.get(_BENCHMARK_PROFILE, {}))

_SQUARE_MATRIX_OPS = {
    "_linalg_det",
    "_linalg_eigh",
    "_linalg_slogdet",
    "inverse",
    "linalg_det",
    "linalg_eig",
    "linalg_eigh",
    "linalg_eigvals",
    "linalg_eigvalsh",
    "linalg_inv",
    "linalg_inv_ex",
    "linalg_ldl_factor",
    "linalg_ldl_factor_ex",
    "linalg_matrix_power",
    "linalg_slogdet",
    "matrix_power",
    "slogdet",
}

_PAD_2D_OPS = {
    "reflection_pad2d",
    "reflection_pad2d_backward",
    "replication_pad2d",
    "replication_pad2d_backward",
}

_PAD_3D_OPS = {
    "reflection_pad3d",
    "reflection_pad3d_backward",
    "replication_pad3d",
    "replication_pad3d_backward",
}

_CONV_2D_OPS = {
    "_conv_depthwise2d",
    "_slow_conv2d_backward",
    "_slow_conv2d_forward",
    "cudnn_convolution",
    "slow_conv_transpose2d",
    "thnn_conv2d",
}

_CONV_3D_OPS = {
    "slow_conv3d",
    "slow_conv3d_forward",
    "slow_conv_transpose3d",
}

_CPU_VALIDATION_ONLY_OPS = {
    "histogram",
}

_ZERO_INITIALIZED_OUTPUT_OPS = {
    "max_unpool2d",
    "max_unpool3d",
}

_POSITIVE_DEFINITE_OPS = {
    "cholesky",
    "cholesky_inverse",
    "linalg_cholesky",
    "linalg_cholesky_ex",
}

_INTEGER_INPUT_OPS = {
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "gcd",
    "lcm",
}

_POSITIVE_DOMAIN_OPS = {
    "cholesky",
    "digamma",
    "igamma",
    "igammac",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "logit",
    "reciprocal",
    "rsqrt",
    "sqrt",
}

_UNIT_INTERVAL_OPS = {
    "acos",
    "asin",
    "atanh",
    "arccos",
    "arcsin",
    "arctanh",
}

_PROBABILITY_OPS = {
    "bernoulli",
    "binary_cross_entropy",
    "binary_cross_entropy_backward",
    "multinomial",
}

_COMPARISON_OPS = {
    "eq",
    "ge",
    "greater",
    "greater_equal",
    "gt",
    "le",
    "less",
    "less_equal",
    "lt",
    "ne",
    "not_equal",
}

_INDEX_PARAMETER_NAMES = {
    "index",
    "indices",
    "block_tables",
    "seq_lens",
    "cache_batch_idx",
}

_CLASS_INDEX_TARGET_OPS = {
    "cross_entropy_loss",
    "multi_margin_loss",
    "multilabel_margin_loss",
    "nll_loss",
    "nll_loss2d",
    "multi_margin_loss_backward",
    "nll_loss_backward",
    "nll_loss_forward",
    "nll_loss2d_backward",
    "nll_loss2d_forward",
}

_NONDETERMINISTIC_OPS = {
    "bernoulli",
    "multinomial",
    "normal",
    "random",
    "random_",
    "rrelu_with_noise",
}

_SCALAR_OVERRIDES = {
    ("hardtanh", "min_val"): -1.0,
    ("hardtanh", "max_val"): 1.0,
    ("clamp", "min"): -1.0,
    ("clamp", "max"): 1.0,
    ("clamp_min", "min"): -1.0,
    ("clamp_max", "max"): 1.0,
    ("threshold", "threshold"): 0.0,
    ("threshold", "value"): 0.0,
    ("leaky_relu", "negative_slope"): 0.01,
    ("softplus", "beta"): 1.0,
    ("softplus", "threshold"): 20.0,
    ("softplus_backward", "beta"): 1.0,
    ("softplus_backward", "threshold"): 20.0,
    ("elu", "alpha"): 1.0,
    ("elu", "scale"): 1.0,
    ("elu", "input_scale"): 1.0,
    ("sub", "alpha"): 1.0,
    ("add", "alpha"): 1.0,
    ("addcdiv", "value"): 1.0,
    ("addcmul", "value"): 1.0,
    ("kthvalue", "k"): 1,
    ("mvlgamma", "p"): 2,
    ("scatter_reduce", "reduce"): "amax",
    ("index_reduce", "reduce"): "amax",
    ("multinomial", "num_samples"): 4,
    ("linalg_qr", "mode"): "reduced",
    ("linalg_solve", "left"): True,
    ("linalg_solve_ex", "left"): True,
    ("linalg_solve_triangular", "left"): True,
    ("linalg_lu_solve", "left"): True,
    ("_linalg_solve_ex", "left"): True,
    ("clip", "min"): -1.0,
    ("clip", "max"): 1.0,
    ("native_batch_norm", "training"): True,
    ("_chunk_cat", "num_chunks"): 2,
    ("_convert_indices_from_coo_to_csr", "size"): 8,
    ("cudnn_convolution", "padding"): [0, 0],
    ("cudnn_convolution", "stride"): [1, 1],
    ("cudnn_convolution", "dilation"): [1, 1],
    ("cudnn_convolution", "groups"): 1,
    ("fft_hfft", "n"): 256,
    ("fft_irfft", "n"): 256,
    ("histc", "bins"): 64,
    ("histc", "min"): -3.0,
    ("histc", "max"): 3.0,
    ("linalg_tensorinv", "ind"): 2,
    ("linalg_matrix_power", "n"): 3,
    ("linalg_matrix_norm", "dim"): [-2, -1],
    ("linalg_lu", "pivot"): True,
    ("matrix_power", "n"): 3,
    ("max_unpool2d", "output_size"): [16, 16],
    ("max_unpool3d", "output_size"): [8, 8, 8],
    ("max_unpool3d", "stride"): [2, 2, 2],
    ("max_unpool3d", "padding"): [0, 0, 0],
    ("linspace", "steps"): 256,
    ("logspace", "steps"): 256,
    ("nanquantile", "interpolation"): "linear",
    ("quantile", "interpolation"): "linear",
    ("special_multigammaln", "p"): 2,
    ("linalg_eigh", "UPLO"): "L",
    ("linalg_eigvalsh", "UPLO"): "L",
    ("_linalg_eigh", "UPLO"): "L",
    ("_linalg_svd", "compute_uv"): True,
}

for _padding_op in _PAD_2D_OPS:
    _SCALAR_OVERRIDES[(_padding_op, "padding")] = [1, 1, 1, 1]
for _padding_op in _PAD_3D_OPS:
    _SCALAR_OVERRIDES[(_padding_op, "padding")] = [1, 1, 1, 1, 1, 1]
for _conv_op in _CONV_2D_OPS:
    _SCALAR_OVERRIDES[(_conv_op, "kernel_size")] = [3, 3]
    _SCALAR_OVERRIDES[(_conv_op, "stride")] = [1, 1]
for _conv_op in _CONV_3D_OPS:
    _SCALAR_OVERRIDES[(_conv_op, "kernel_size")] = [3, 3, 3]
    _SCALAR_OVERRIDES[(_conv_op, "stride")] = [1, 1, 1]

_REFERENCE_KEYWORD_PARAMS = {
    "alpha",
    "beta",
    "correction",
    "device",
    "dtype",
    "generator",
    "layout",
    "memory_format",
    "pin_memory",
    "sparse_grad",
    "approximate",
}


class RecipeError(RuntimeError):
    """Raised when the first-version recipe cannot safely drive a schema."""


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def canonical_schemas(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one deterministic schema per generated wrapper name."""

    selected: dict[str, dict[str, Any]] = {}
    for schema in metadata.get("ops", []):
        name = schema.get("name")
        if name and name not in selected:
            selected[name] = schema
    return [selected[name] for name in sorted(selected)]


def _list_size(aten_type: str) -> int:
    if "[" not in aten_type or "]" not in aten_type:
        return 1
    text = aten_type.split("[", 1)[1].split("]", 1)[0]
    return int(text) if text.isdigit() else 1


def scalar_value(op_name: str, param: dict[str, Any]) -> Any:
    """Choose a fixed, conservative value for a non-tensor parameter."""

    name = param.get("name", "")
    aten_type = param.get("type", "")
    override = _SCALAR_OVERRIDES.get((op_name, name))
    if override is not None:
        return override

    if aten_type.endswith("?") or aten_type in {
        "Tensor?",
        "ScalarType?",
        "Generator?",
        "MemoryFormat?",
        "Layout?",
        "Device?",
        "str?",
        "float[]?",
        "int[]?",
        "SymInt[]?",
    }:
        return None

    if name in {"dim", "axis"}:
        if aten_type.startswith(("int[", "SymInt[")) or aten_type in {
            "int[]",
            "SymInt[]",
        }:
            return [-1]
        return -1
    if name == "k":
        return 1
    if name in {"kernel_size", "stride"}:
        return [2] * _list_size(aten_type)
    if name in {"padding", "output_padding"}:
        return [0] * _list_size(aten_type)
    if name == "dilation":
        return [1] * _list_size(aten_type)
    if name == "output_size":
        return [4] * _list_size(aten_type)
    if name == "input_size" and "upsample" in op_name:
        if "1d" in op_name:
            return [2, 4, 16]
        if "2d" in op_name:
            return [2, 4, 16, 16]
        if "3d" in op_name:
            return [2, 2, 8, 8, 8]
    if name in {"dims", "dim_list"}:
        return [-1]
    if name == "reduction":
        return 1
    if name in {"p", "ord"}:
        return 2.0
    if name in {"keepdim", "upper", "transpose", "conjugate", "stable"}:
        return False
    if name in {"sorted", "largest"}:
        return True

    base_type = aten_type.rstrip("?")
    if base_type.startswith(("int[", "SymInt[")) or base_type in {
        "int[]",
        "SymInt[]",
    }:
        return [0] * _list_size(base_type)
    if base_type in {"int", "SymInt"}:
        return 0
    if base_type == "bool":
        return False
    if base_type in {"float", "Scalar"}:
        return 0.5
    if base_type == "str":
        return "none"
    if base_type in {"ScalarType", "MemoryFormat", "Layout", "Device"}:
        return None

    raise RecipeError(f"no fixed scalar recipe for {name}:{aten_type}")


def recipe_for_tensor(
    op_name: str,
    param_name: str,
    tensor_index: int,
    default_dtype: str = "float16",
) -> dict[str, Any]:
    """Describe a deterministic tensor without importing Torch."""

    shape = list(_DEFAULT_SHAPE)
    lowered = param_name.lower()
    matrix_shapes = _MATRIX_SHAPES.get(op_name)
    if matrix_shapes is not None and tensor_index < len(matrix_shapes):
        shape = list(matrix_shapes[tensor_index])

    if "upsample" in op_name:
        if "1d" in op_name:
            shape = [2, 4, 4] if "backward" in op_name and tensor_index == 0 else [2, 4, 16]
        elif "2d" in op_name:
            shape = (
                [2, 4, 4, 4]
                if "backward" in op_name and tensor_index == 0
                else [2, 4, 16, 16]
            )
        elif "3d" in op_name:
            shape = (
                [2, 2, 4, 4, 4]
                if "backward" in op_name and tensor_index == 0
                else [2, 2, 8, 8, 8]
            )
    elif "pool2d" in op_name:
        shape = [2, 8, 16, 16]
        if "backward" in op_name and tensor_index == 0:
            shape = [2, 8, 8, 8]
    elif "pool3d" in op_name:
        shape = [2, 4, 8, 8, 8]
        if "backward" in op_name and tensor_index == 0:
            shape = [2, 4, 4, 4, 4]
    elif op_name in _PAD_2D_OPS:
        shape = [2, 4, 18, 18] if "backward" in op_name and tensor_index == 0 else [2, 4, 16, 16]
    elif op_name in _PAD_3D_OPS:
        shape = (
            [2, 4, 10, 10, 10]
            if "backward" in op_name and tensor_index == 0
            else [2, 4, 8, 8, 8]
        )
    elif op_name in _CONV_2D_OPS:
        if param_name == "weight":
            shape = [8, 1, 3, 3] if op_name == "_conv_depthwise2d" else [16, 8, 3, 3]
            if op_name == "slow_conv_transpose2d":
                shape = [8, 16, 3, 3]
        elif param_name == "grad_output":
            shape = [2, 16, 14, 14]
        else:
            shape = [2, 8, 16, 16]
    elif op_name in _CONV_3D_OPS:
        if param_name == "weight":
            shape = (
                [8, 16, 3, 3, 3]
                if op_name == "slow_conv_transpose3d"
                else [16, 8, 3, 3, 3]
            )
        else:
            shape = [2, 8, 8, 8, 8]
    elif op_name == "im2col":
        shape = [2, 8, 16, 16]
    elif op_name == "col2im":
        shape = [2, 32, 4]
    elif op_name in {"cross", "linalg_cross"}:
        shape = [4096, 3]
    elif op_name in {"fft_hfft", "fft_hfft2", "fft_hfftn", "fft_irfft", "fft_irfft2", "fft_irfftn"}:
        shape = [8, 129]
    elif op_name in {"linspace", "logspace"}:
        shape = []
    elif op_name == "linalg_tensorinv":
        shape = [4, 4, 4, 4]
    elif op_name in {"quantile", "nanquantile"} and param_name == "q":
        return {"shape": [3], "dtype": default_dtype, "kind": "probability"}
    elif op_name == "bucketize" and param_name == "boundaries":
        return {"shape": [64], "dtype": default_dtype, "kind": "sorted"}
    elif op_name == "histogram" and param_name == "bins":
        return {"shape": [65], "dtype": default_dtype, "kind": "sorted"}
    elif op_name == "_convert_indices_from_coo_to_csr":
        return {"shape": [32], "dtype": "int64", "kind": "sorted_index", "high": 8}
    elif op_name == "_batch_norm_with_update":
        if param_name in {"running_mean", "running_var"}:
            kind = "positive" if param_name == "running_var" else "normal"
            return {"shape": [8], "dtype": "float32", "kind": kind}
        shape = [4, 8, 8, 8]
    elif op_name in {"_softmax_backward_data", "_log_softmax_backward_data"}:
        shape = [8, 128]
    elif op_name == "glu_backward":
        shape = [8, 64] if param_name == "grad_output" else [8, 128]
    elif op_name in {"nll_loss2d", "nll_loss2d_forward", "nll_loss2d_backward"}:
        if param_name == "input":
            shape = [2, 16, 8, 8]
        elif param_name == "target":
            return {
                "shape": [2, 8, 8],
                "dtype": "int64",
                "kind": "bounded_index",
                "high": 16,
            }
        elif param_name in {"grad_output", "total_weight"}:
            return {"shape": [], "dtype": default_dtype, "kind": "positive"}
    elif op_name in {"nll_loss_forward", "nll_loss_backward"}:
        if param_name == "input":
            shape = [8, 128]
        elif param_name == "target":
            return {
                "shape": [8],
                "dtype": "int64",
                "kind": "bounded_index",
                "high": 128,
            }
        elif param_name in {"grad_output", "total_weight"}:
            return {"shape": [], "dtype": default_dtype, "kind": "positive"}
    elif op_name == "multi_margin_loss_backward":
        if param_name == "grad_output":
            return {"shape": [], "dtype": default_dtype, "kind": "normal"}
        if param_name == "input":
            shape = [8, 128]
        elif param_name == "target":
            return {
                "shape": [8],
                "dtype": "int64",
                "kind": "bounded_index",
                "high": 128,
            }
    elif op_name.startswith("multilabel_margin_loss"):
        if param_name == "grad_output":
            return {"shape": [], "dtype": default_dtype, "kind": "normal"}
        if param_name == "target":
            return {
                "shape": [8, 32],
                "dtype": "int64",
                "kind": "multilabel_target",
            }
        if param_name == "is_target":
            return {"shape": [8, 32], "dtype": "int64", "kind": "integer"}
        shape = [8, 32]

    if op_name in _SQUARE_MATRIX_OPS and tensor_index == 0:
        return {
            "shape": [32, 32],
            "dtype": default_dtype,
            "kind": "positive_definite",
        }

    if op_name == "_linalg_solve_ex":
        if param_name == "A":
            return {
                "shape": [32, 32],
                "dtype": default_dtype,
                "kind": "positive_definite",
            }
        if param_name == "B":
            return {"shape": [32, 8], "dtype": default_dtype, "kind": "normal"}

    if op_name == "linalg_ldl_solve":
        if param_name == "B":
            return {"shape": [32, 8], "dtype": default_dtype, "kind": "normal"}
        if param_name == "pivots":
            return {"shape": [32], "dtype": "int32", "kind": "integer"}
        return {
            "shape": [32, 32],
            "dtype": default_dtype,
            "kind": "positive_definite",
        }

    if op_name in {"linalg_lu_solve", "lu_solve"} and "pivot" in lowered:
        return {"shape": [32], "dtype": "int32", "kind": "integer"}

    if op_name == "cholesky_solve" and param_name == "input2":
        return {
            "shape": [32, 32],
            "dtype": default_dtype,
            "kind": "lower_triangular",
        }

    if op_name == "_int_mm":
        return {"shape": shape, "dtype": "int8", "kind": "integer"}

    if op_name == "where" and lowered == "condition":
        return {"shape": shape, "dtype": "bool", "kind": "boolean"}

    if op_name in _POSITIVE_DEFINITE_OPS and tensor_index == 0:
        return {
            "shape": [32, 32],
            "dtype": default_dtype,
            "kind": "positive_definite",
        }

    if op_name == "triangular_solve" and lowered == "a":
        return {"shape": shape, "dtype": default_dtype, "kind": "upper_triangular"}

    if op_name in {"linalg_solve", "linalg_solve_ex", "linalg_solve_triangular"}:
        if lowered in {"a", "input"}:
            kind = "upper_triangular" if "triangular" in op_name else "positive_definite"
            return {"shape": shape, "dtype": default_dtype, "kind": kind}

    if lowered == "target" and op_name in _CLASS_INDEX_TARGET_OPS:
        return {
            "shape": [8],
            "dtype": "int64",
            "kind": "bounded_index",
            "high": 128,
        }

    if lowered in _INDEX_PARAMETER_NAMES or any(
        token in lowered for token in ("index", "indices")
    ):
        index_shape = [8, 128]
        if op_name.startswith("index_"):
            index_shape = [4]
        kind = "unique_index" if op_name.startswith("scatter") else "bounded_index"
        return {
            "shape": index_shape,
            "dtype": "int64",
            "kind": kind,
            "high": 128,
        }

    if "mask" in lowered:
        return {"shape": shape, "dtype": "bool", "kind": "boolean"}

    if op_name.startswith("index_") and lowered in {"source", "src"}:
        return {"shape": [8, 4], "dtype": default_dtype, "kind": "normal"}

    if op_name == "searchsorted" and lowered == "sorted_sequence":
        return {"shape": shape, "dtype": default_dtype, "kind": "sorted"}

    if op_name in _INTEGER_INPUT_OPS:
        return {"shape": shape, "dtype": "int32", "kind": "integer"}
    if op_name in _PROBABILITY_OPS:
        return {"shape": shape, "dtype": default_dtype, "kind": "probability"}
    if op_name == "angle":
        return {"shape": shape, "dtype": "float32", "kind": "normal"}
    if op_name in _UNIT_INTERVAL_OPS:
        return {"shape": shape, "dtype": default_dtype, "kind": "unit_interval"}
    if op_name in _POSITIVE_DOMAIN_OPS:
        return {"shape": shape, "dtype": default_dtype, "kind": "positive"}

    return {"shape": shape, "dtype": default_dtype, "kind": "normal"}


def recipe_for_tensor_list(
    op_name: str,
    param_name: str,
    default_dtype: str = "float16",
) -> dict[str, Any]:
    """Describe a deterministic list of compatible tensors."""

    if op_name in {"chain_matmul", "linalg_multi_dot"}:
        shapes = [[32, 48], [48, 24], [24, 16]]
    elif op_name == "cat":
        shapes = [list(_DEFAULT_SHAPE), list(_DEFAULT_SHAPE)]
    elif op_name == "index" and "indices" in param_name:
        return {
            "shapes": [[4]],
            "dtype": "int64",
            "kind": "index_tensor_list",
            "high": 8,
        }
    else:
        shapes = [[8, 64], [8, 64]]
    return {
        "shapes": shapes,
        "dtype": default_dtype,
        "kind": "tensor_list",
    }


def _recipe_text(named_recipes: list[tuple[str, dict[str, Any]]]) -> str:
    parts = []
    for name, recipe in named_recipes:
        if "shapes" in recipe:
            shapes = ",".join(
                "x".join(str(value) for value in shape) for shape in recipe["shapes"]
            )
            parts.append(f"{name}={recipe['dtype']}[{shapes}]:{recipe['kind']}")
        else:
            shape = "x".join(str(value) for value in recipe["shape"])
            parts.append(f"{name}={recipe['dtype']}[{shape}]:{recipe['kind']}")
    return "; ".join(parts)


def split_reference_arguments(
    params: list[dict[str, Any]], values: list[Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Route known Python keyword-only ATen parameters by name."""

    positional = []
    keyword = {}
    for param, value in zip(params, values):
        if param.get(
            "keyword_only", param.get("name") in _REFERENCE_KEYWORD_PARAMS
        ):
            keyword[param["name"]] = value
        else:
            positional.append(value)
    return positional, keyword


def schema_parameter_groups(
    schema: dict[str, Any],
) -> tuple[bool, bool, list[dict[str, Any]], list[dict[str, Any]]]:
    """Split inputs/outputs while preserving explicit-out inplace wrappers."""

    aten_name = schema.get("aten_name", schema["name"])
    is_inplace = aten_name.endswith("_") and not aten_name.endswith("__")
    params = schema.get("params", [])
    out_params = [param for param in params if param.get("is_out")]
    is_implicit_inplace = is_inplace and not out_params
    in_params = params if is_implicit_inplace else [
        param for param in params if not param.get("is_out")
    ]
    return is_inplace, is_implicit_inplace, in_params, out_params


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_status = Counter(row.get("status", "unknown") for row in rows)
    successful = [row for row in rows if row.get("status") == "success"]
    latencies = [float(row["infini_us"]) for row in successful]
    return {
        "attempted": len(rows),
        "success": len(successful),
        "by_status": dict(sorted(by_status.items())),
        "coverage_pct": round(100.0 * len(successful) / len(rows), 2) if rows else 0.0,
        "median_infini_us": statistics.median(latencies) if latencies else None,
    }


def build_worker_command(
    *,
    python: str,
    script: str,
    metadata: str,
    op_name: str,
    device: str,
    dtype: str,
    min_run_time: float,
) -> list[str]:
    return [
        python,
        script,
        "--worker",
        "--metadata",
        metadata,
        "--worker-op",
        op_name,
        "--device",
        device,
        "--dtype",
        dtype,
        "--min-run-time",
        str(min_run_time),
    ]


def build_worker_batch_command(
    *,
    python: str,
    script: str,
    metadata: str,
    op_names: list[str],
    device: str,
    dtype: str,
    min_run_time: float,
) -> list[str]:
    return [
        python,
        script,
        "--worker",
        "--metadata",
        metadata,
        "--worker-ops",
        ",".join(op_names),
        "--device",
        device,
        "--dtype",
        dtype,
        "--min-run-time",
        str(min_run_time),
    ]


def parse_worker_stdout(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("record_type") == "result":
            return value
    return None


def parse_worker_stdout_rows(stdout: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("record_type") == "result" and value.get("op"):
            rows[str(value["op"])] = value
    return rows


def _clean_error(value: Any, limit: int = 800) -> str:
    text = " ".join(str(value).split())
    return text[:limit]


def write_reports(
    output_dir: Path | str,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(rows)
    payload = {"metadata": metadata, "summary": summary, "results": rows}
    (output / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            for key in ("active_implementation_indices",):
                if isinstance(serialized.get(key), list):
                    serialized[key] = json.dumps(serialized[key])
            writer.writerow(serialized)

    lines = [
        "# InfiniOps full operator benchmark",
        "",
        f"- SHA: `{metadata.get('sha', 'unknown')}`",
        f"- Device: `{metadata.get('device', 'unknown')}`",
        f"- Dtype request: `{metadata.get('dtype', 'unknown')}`",
        f"- Attempted wrappers: {summary['attempted']}",
        f"- Successful benchmarks: {summary['success']} ({summary['coverage_pct']:.2f}%)",
        "- Timing: `torch.utils.benchmark.Timer.blocked_autorange().median`, unit `us`",
        "",
        "## Status counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status, count in summary["by_status"].items():
        lines.append(f"| `{status}` | {count} |")

    failures = [row for row in rows if row.get("status") != "success"]
    lines.extend(["", "## Non-success details", ""])
    if failures:
        lines.extend(["| Operator | Status | Reason |", "|---|---|---|"])
        for row in failures:
            reason = str(row.get("error", "")).replace("|", "\\|")
            lines.append(f"| `{row.get('op', '')}` | `{row.get('status', '')}` | {reason} |")
    else:
        lines.append("All attempted wrappers succeeded.")

    lines.extend(
        [
            "",
            "## Audit",
            "",
            f"- Start: `{metadata.get('start', '')}`",
            f"- End: `{metadata.get('end', '')}`",
            f"- Command: `{metadata.get('command', '')}`",
            f"- Metadata: `{metadata.get('metadata_path', '')}`",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pascal(name: str) -> str:
    return "".join(part.capitalize() for part in name.strip("_").split("_"))


def _torch_dtype(torch: Any, name: str) -> Any:
    value = getattr(torch, name, None)
    if value is None:
        raise RecipeError(f"torch has no dtype {name}")
    return value


def _make_tensor(torch: Any, recipe: dict[str, Any], device: str) -> Any:
    dtype = _torch_dtype(torch, recipe["dtype"])
    kind = recipe["kind"]
    if kind in {"tensor_list", "index_tensor_list"}:
        tensors = []
        for shape in recipe["shapes"]:
            if kind == "index_tensor_list":
                tensor = torch.randint(
                    0, recipe["high"], shape, dtype=dtype, device=device
                )
            else:
                tensor = torch.randn(shape, dtype=dtype, device=device)
            tensors.append(tensor)
        return tensors

    shape = recipe["shape"]
    if kind == "bounded_index":
        return torch.randint(0, recipe["high"], shape, dtype=dtype, device=device)
    if kind == "unique_index":
        values = torch.arange(shape[-1], dtype=dtype, device=device) % recipe["high"]
        view_shape = [1] * (len(shape) - 1) + [shape[-1]]
        return values.view(view_shape).expand(shape).clone()
    if kind == "sorted_index":
        return torch.randint(0, recipe["high"], shape, dtype=dtype, device=device).sort().values
    if kind == "multilabel_target":
        target = torch.full(shape, -1, dtype=dtype, device=device)
        count = min(4, shape[-1])
        target[..., :count] = torch.arange(count, dtype=dtype, device=device)
        return target
    if kind == "integer":
        return torch.randint(1, 16, shape, dtype=dtype, device=device)
    if kind == "boolean":
        return torch.rand(shape, device=device) > 0.5
    if kind == "positive":
        return torch.rand(shape, dtype=dtype, device=device) + 0.5
    if kind == "unit_interval":
        return torch.rand(shape, dtype=dtype, device=device) * 1.8 - 0.9
    if kind == "probability":
        return torch.rand(shape, dtype=dtype, device=device) * 0.8 + 0.1
    if kind == "sorted":
        return torch.randn(shape, dtype=dtype, device=device).sort(dim=-1).values
    if kind == "positive_definite":
        source = torch.randn(shape, dtype=dtype, device=device)
        identity = torch.eye(shape[-1], dtype=dtype, device=device)
        return source @ source.transpose(-1, -2) + identity
    if kind == "upper_triangular":
        source = torch.randn(shape, dtype=dtype, device=device)
        return torch.triu(source) + torch.eye(shape[-1], dtype=dtype, device=device)
    if kind == "lower_triangular":
        source = torch.randn(shape, dtype=dtype, device=device)
        return torch.tril(source) + torch.eye(shape[-1], dtype=dtype, device=device)
    return torch.randn(shape, dtype=dtype, device=device)


def _relate_inputs(
    torch: Any,
    aten_name: str,
    params: list[dict[str, Any]],
    inputs: list[Any],
    device: str,
    dtype_name: str,
) -> tuple[list[Any], str]:
    """Replace independently generated values when an ATen schema links inputs."""

    positions = {param["name"]: index for index, param in enumerate(params)}

    def has(*names: str) -> bool:
        return all(name in positions for name in names)

    def get(name: str) -> Any:
        return inputs[positions[name]]

    def put(name: str, value: Any) -> None:
        inputs[positions[name]] = value

    note = ""
    if aten_name in {"_softmax_backward_data", "_log_softmax_backward_data"}:
        source = torch.randn_like(get("output"))
        output = (
            torch.log_softmax(source, dim=-1)
            if aten_name == "_log_softmax_backward_data"
            else torch.softmax(source, dim=-1)
        )
        put("output", output)
        put("input_dtype", _torch_dtype(torch, dtype_name))
        note = "forward-derived softmax output and explicit input dtype"

    elif aten_name in {"adaptive_max_pool2d_backward", "adaptive_max_pool3d_backward"}:
        input_value = get("input")
        output_size = (8, 8) if "2d" in aten_name else (4, 4, 4)
        pool = (
            torch.nn.functional.adaptive_max_pool2d
            if "2d" in aten_name
            else torch.nn.functional.adaptive_max_pool3d
        )
        pooled, indices = pool(input_value, output_size, return_indices=True)
        put("grad_output", torch.randn_like(pooled))
        put("indices", indices)
        note = "indices and grad_output derived from adaptive max-pool forward"

    elif aten_name in {
        "fractional_max_pool2d_backward",
        "fractional_max_pool3d_backward",
    }:
        input_value = get("input")
        pool = (
            torch.nn.functional.fractional_max_pool2d
            if "2d" in aten_name
            else torch.nn.functional.fractional_max_pool3d
        )
        pooled, indices = pool(
            input_value,
            get("kernel_size"),
            output_size=get("output_size"),
            return_indices=True,
        )
        put("grad_output", torch.randn_like(pooled))
        put("indices", indices)
        note = "indices and grad_output derived from fractional max-pool forward"

    elif aten_name in {
        "max_pool2d_with_indices_backward",
        "max_pool3d_with_indices_backward",
    }:
        pool = (
            torch.nn.functional.max_pool2d
            if "2d" in aten_name
            else torch.nn.functional.max_pool3d
        )
        pooled, indices = pool(
            get("input"),
            get("kernel_size"),
            get("stride"),
            get("padding"),
            get("dilation"),
            get("ceil_mode"),
            return_indices=True,
        )
        put("grad_output", torch.randn_like(pooled))
        put("indices", indices)
        note = "indices and grad_output derived from max-pool forward"

    elif aten_name in {"max_unpool2d", "max_unpool3d"}:
        source_shape = [2, 8, 16, 16] if "2d" in aten_name else [2, 4, 8, 8, 8]
        source = torch.randn(
            source_shape,
            dtype=_torch_dtype(torch, dtype_name),
            device=device,
        )
        pool = (
            torch.nn.functional.max_pool2d
            if "2d" in aten_name
            else torch.nn.functional.max_pool3d
        )
        pooled, indices = pool(source, 2, 2, return_indices=True)
        put("input", pooled)
        put("indices", indices)
        note = "pooled values and indices derived from max-pool forward"

    elif aten_name in {
        "reflection_pad2d_backward",
        "reflection_pad3d_backward",
        "replication_pad2d_backward",
        "replication_pad3d_backward",
    }:
        mode = "reflect" if aten_name.startswith("reflection") else "replicate"
        padded = torch.nn.functional.pad(get("input"), get("padding"), mode=mode)
        put("grad_output", torch.randn_like(padded))
        note = f"grad_output shaped from {mode} padding forward"

    elif aten_name == "_slow_conv2d_backward":
        output = torch.nn.functional.conv2d(
            get("input"),
            get("weight"),
            stride=get("stride"),
            padding=get("padding"),
        )
        put("grad_output", torch.randn_like(output))
        note = "grad_output shaped from conv2d forward"

    elif aten_name == "linalg_ldl_solve" and has("LD", "pivots"):
        target_dtype = get("LD").dtype
        factor_dtype = (
            torch.float32
            if target_dtype in {torch.float16, torch.bfloat16}
            else target_dtype
        )
        source = torch.randn([32, 32], dtype=factor_dtype, device=device)
        source = source @ source.transpose(-1, -2) + torch.eye(
            32, dtype=source.dtype, device=device
        )
        ld, pivots = torch.linalg.ldl_factor(source)
        put("LD", ld.to(target_dtype))
        put("pivots", pivots)
        note = "LD and pivots derived from ldl_factor"

    elif aten_name in {"linalg_lu_solve", "lu_solve"}:
        target_dtype = get("LU").dtype if has("LU") else get("LU_data").dtype
        factor_dtype = (
            torch.float32
            if target_dtype in {torch.float16, torch.bfloat16}
            else target_dtype
        )
        source = torch.randn([32, 32], dtype=factor_dtype, device=device)
        source = source + torch.eye(32, dtype=factor_dtype, device=device)
        lu, pivots = torch.linalg.lu_factor(source)
        if has("LU"):
            put("LU", lu.to(target_dtype))
            put("pivots", pivots)
        else:
            put("LU_data", lu.to(target_dtype))
            put("LU_pivots", pivots)
        note = "LU and pivots derived from lu_factor"

    elif aten_name == "multilabel_margin_loss_backward" and has("is_target"):
        _, is_target = torch.ops.aten.multilabel_margin_loss_forward.default(
            get("input"),
            get("target"),
            get("reduction"),
        )
        put("is_target", is_target)
        note = "is_target derived from multilabel margin-loss forward"

    return inputs, note


def _clone_value(torch: Any, value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return [_clone_value(torch, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(torch, item) for item in value)
    return value


def _first_tensor(torch: Any, values: list[Any]) -> Any | None:
    for value in values:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (list, tuple)):
            nested = _first_tensor(torch, list(value))
            if nested is not None:
                return nested
    return None


def _resolve_aten_reference(
    torch: Any,
    aten_name: str,
    params: list[dict[str, Any]],
) -> tuple[Any | None, list[dict[str, Any]]]:
    """Resolve the matching ATen overload and copy its kwarg-only layout."""

    packet = getattr(torch.ops.aten, aten_name, None)
    if packet is None or not hasattr(packet, "overloads"):
        return None, params

    candidates = []
    for overload_name in packet.overloads():
        overload = getattr(packet, overload_name)
        arguments = list(overload._schema.arguments)
        if len(arguments) != len(params):
            continue
        score = 0
        for param, argument in zip(params, arguments):
            param_name = str(param.get("name", "")).lower()
            argument_name = str(argument.name).lower()
            if param_name == argument_name or {param_name, argument_name} <= {
                "input",
                "self",
            }:
                score += 2
            param_type = str(param.get("type", "")).replace("SymInt", "int")
            argument_type = str(argument.type).replace("SymInt", "int")
            if param_type.rstrip("?").split("(", 1)[0] in argument_type:
                score += 1
        candidates.append((score, overload_name == "out", overload, arguments))

    if not candidates:
        return None, params

    _, _, overload, arguments = max(
        candidates,
        key=lambda item: (item[0], not item[1]),
    )
    annotated = []
    for param, argument in zip(params, arguments):
        value = dict(param)
        value["keyword_only"] = bool(argument.kwarg_only)
        annotated.append(value)
    return overload, annotated


def _reference_function(torch: Any, aten_name: str) -> Any:
    if aten_name == "_add_relu":
        return lambda input, other, alpha=1: torch.relu(input + alpha * other)
    if aten_name == "adaptive_avg_pool3d_backward":
        return torch.ops.aten._adaptive_avg_pool3d_backward.default
    if aten_name == "histogram":
        def histogram_cpu(input: Any, bins: Any, weight: Any, density: bool) -> Any:
            cpu_weight = None if weight is None else weight.cpu().float()
            hist, edges = torch.histogram(
                input.cpu().float(),
                bins=bins.cpu().float(),
                weight=cpu_weight,
                density=density,
            )
            return (
                hist.to(device=input.device, dtype=input.dtype),
                edges.to(device=input.device, dtype=input.dtype),
            )

        return histogram_cpu
    if aten_name == "mkldnn_adaptive_avg_pool2d":
        return torch.nn.functional.adaptive_avg_pool2d
    if aten_name in {"slow_conv3d", "slow_conv3d_forward"}:
        return lambda input, weight, kernel_size, bias, stride, padding: (
            torch.nn.functional.conv3d(
                input,
                weight,
                bias=bias,
                stride=stride,
                padding=padding,
            )
        )
    if aten_name == "_slow_conv2d_backward":
        return lambda grad_output, input, weight, kernel_size, stride, padding: (
            torch.ops.aten._slow_conv2d_backward.output_mask(
                grad_output,
                input,
                weight,
                kernel_size,
                stride,
                padding,
                [True, True, True],
            )
        )
    if aten_name.endswith("_") and not aten_name.endswith("__"):
        return lambda input, *args, **kwargs: getattr(input, aten_name)(*args, **kwargs)

    candidates = [
        getattr(torch, aten_name, None),
        getattr(torch.special, aten_name, None),
        getattr(torch.nn.functional, aten_name, None),
    ]
    if aten_name.startswith("special_"):
        candidates.append(getattr(torch.special, aten_name.removeprefix("special_"), None))
    for candidate in candidates:
        if candidate is not None:
            return candidate
    aten_candidate = getattr(torch.ops.aten, aten_name, None)
    if aten_candidate is not None:
        return aten_candidate
    raise RecipeError(f"no PyTorch reference function for {aten_name}")


def _supported_infini_dtype(torch: Any, tensor: Any) -> bool:
    names = (
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
    )
    return tensor.dtype in {getattr(torch, name) for name in names if hasattr(torch, name)}


def _assert_close(torch: Any, actual: Any, expected: Any, dtype_name: str) -> None:
    if actual.dtype.is_floating_point:
        tolerance = 1e-2 if dtype_name in {"float16", "bfloat16"} else 1e-5
        if not torch.allclose(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True):
            delta = (actual.float() - expected.float()).abs().max().item()
            raise AssertionError(f"max absolute difference {delta}")
    elif not torch.equal(actual, expected):
        raise AssertionError("integer outputs differ")


def _measure_us(torch: Any, fn: Any, min_run_time: float, device: str) -> float:
    device_type = _device_type(device)
    if device_type in {"cpu", "cuda"}:
        timer = torch.utils.benchmark.Timer(stmt="fn()", globals={"fn": fn})
        return timer.blocked_autorange(min_run_time=min_run_time).median * 1e6

    for _ in range(5):
        fn()
    _synchronize(torch, device)

    target_seconds = max(float(min_run_time), 0.001)
    iterations = 1
    while True:
        _synchronize(torch, device)
        started = time.perf_counter()
        for _ in range(iterations):
            fn()
        _synchronize(torch, device)
        elapsed = time.perf_counter() - started
        if elapsed >= target_seconds or iterations >= 1_000_000:
            break
        growth = max(2, min(10, math.ceil(target_seconds / max(elapsed, 1e-9))))
        iterations = min(1_000_000, iterations * growth)

    samples = []
    for _ in range(5):
        _synchronize(torch, device)
        started = time.perf_counter()
        for _ in range(iterations):
            fn()
        _synchronize(torch, device)
        samples.append((time.perf_counter() - started) / iterations)
    return statistics.median(samples) * 1e6


def _base_result(schema: dict[str, Any], requested_dtype: str) -> dict[str, Any]:
    return {
        "record_type": "result",
        "op": schema.get("name", ""),
        "aten_name": schema.get("aten_name", schema.get("name", "")),
        "overload_name": schema.get("overload_name", ""),
        "status": "worker_error",
        "implementation_index": None,
        "active_implementation_indices": [],
        "benchmark_profile": _BENCHMARK_PROFILE,
        "dtype": requested_dtype,
        "input_recipe": "",
        "infini_us": None,
        "ref_us": None,
        "speedup": None,
        "ref_mode": "functional",
        "validation_mode": "exact",
        "error": "",
    }


def run_worker(args: argparse.Namespace) -> int:
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    schemas_by_name = {
        schema["name"]: schema for schema in canonical_schemas(metadata)
    }
    if args.worker_ops:
        requested_names = [
            name.strip() for name in args.worker_ops.split(",") if name.strip()
        ]
    else:
        requested_names = [args.worker_op or ""]

    return_code = 0
    try:
        import torch
        import torch.utils.benchmark

        _load_device_plugin(args.device)
        import infini.ops

        torch.manual_seed(20260730)
        _manual_seed_all(torch, args.device, 20260730)
    except Exception as exc:  # noqa: BLE001
        for op_name in requested_names:
            result = {
                "record_type": "result",
                "op": op_name,
                "status": "worker_error",
                "error": _clean_error(f"{type(exc).__name__}: {exc}"),
            }
            print(json.dumps(result, ensure_ascii=False), flush=True)
        return 2

    for op_name in requested_names:
        schema = schemas_by_name.get(op_name)
        if schema is None:
            print(
                json.dumps(
                    {
                        "record_type": "result",
                        "op": op_name,
                        "status": "metadata_missing",
                        "error": "wrapper not found in metadata",
                    }
                ),
                flush=True,
            )
            return_code = max(return_code, 2)
            continue

        result = _base_result(schema, args.dtype)
        started = time.monotonic()
        try:
            torch.manual_seed(20260730)
            _manual_seed_all(torch, args.device, 20260730)

            op_class = getattr(infini.ops, _pascal(schema["name"]), None)
            if op_class is None:
                result.update(status="backend_unsupported", error="wrapper class is not exposed")
                op_return_code = 0
            else:
                active = list(op_class.active_implementation_indices(args.device))
                result["active_implementation_indices"] = active
                if not active:
                    result.update(
                        status="backend_unsupported",
                        error=f"no active implementation on {args.device}",
                    )
                    op_return_code = 0
                else:
                    implementation_index = active[0]
                    result["implementation_index"] = implementation_index
                    op_return_code = _run_schema_case(
                        torch=torch,
                        infini_ops=infini.ops,
                        schema=schema,
                        result=result,
                        device=args.device,
                        dtype_name=args.dtype,
                        min_run_time=args.min_run_time,
                        implementation_index=implementation_index,
                    )
        except RecipeError as exc:
            result.update(status="recipe_unsupported", error=_clean_error(exc))
            op_return_code = 0
        except Exception as exc:  # noqa: BLE001
            result.update(
                status="worker_error",
                error=_clean_error(f"{type(exc).__name__}: {exc}"),
                traceback=_clean_error(traceback.format_exc(), 2000),
            )
            op_return_code = 0
        finally:
            result["duration_s"] = round(time.monotonic() - started, 3)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        return_code = max(return_code, op_return_code)
    return return_code


def _run_schema_case(
    *,
    torch: Any,
    infini_ops: Any,
    schema: dict[str, Any],
    result: dict[str, Any],
    device: str,
    dtype_name: str,
    min_run_time: float,
    implementation_index: int,
) -> int:
    aten_name = schema.get("aten_name", schema["name"])
    is_inplace, is_implicit_inplace, in_params, out_params = schema_parameter_groups(
        schema
    )

    inputs = []
    named_recipes = []
    tensor_index = 0
    for param in in_params:
        param_type = param.get("type", "")
        if "Tensor" in param_type and "[]" in param_type:
            recipe = recipe_for_tensor_list(
                aten_name, param["name"], default_dtype=dtype_name
            )
            inputs.append(_make_tensor(torch, recipe, device))
            named_recipes.append((param["name"], recipe))
        elif param.get("is_tensor"):
            if "?" in param.get("type", ""):
                inputs.append(None)
                continue
            recipe = recipe_for_tensor(aten_name, param["name"], tensor_index, dtype_name)
            inputs.append(_make_tensor(torch, recipe, device))
            named_recipes.append((param["name"], recipe))
            tensor_index += 1
        else:
            inputs.append(scalar_value(aten_name, param))

    inputs, relation_note = _relate_inputs(
        torch,
        aten_name,
        in_params,
        inputs,
        device,
        dtype_name,
    )
    result["input_recipe"] = _recipe_text(named_recipes)
    if relation_note:
        result["input_recipe"] += f"; related={relation_note}"
    first_tensor = _first_tensor(torch, inputs)
    effective_dtype = (
        first_tensor.dtype if first_tensor is not None else _torch_dtype(torch, dtype_name)
    )
    result["dtype"] = str(effective_dtype).removeprefix("torch.")

    reference = _reference_function(torch, aten_name)
    reference_inputs = [_clone_value(torch, value) for value in inputs]
    aten_reference, reference_params = _resolve_aten_reference(
        torch, aten_name, in_params
    )
    reference_args, reference_kwargs = split_reference_arguments(
        reference_params, reference_inputs
    )
    try:
        reference_value = reference(*reference_args, **reference_kwargs)
    except Exception as exc:  # noqa: BLE001
        if aten_reference is None or aten_reference is reference:
            result.update(
                status="reference_rejected",
                error=_clean_error(f"{type(exc).__name__}: {exc}"),
            )
            return 0
        try:
            reference = aten_reference
            reference_value = reference(*reference_args, **reference_kwargs)
            result["ref_mode"] = "aten_functional"
        except Exception as aten_exc:  # noqa: BLE001
            result.update(
                status="reference_rejected",
                error=_clean_error(
                    f"{type(exc).__name__}: {exc}; ATen fallback: "
                    f"{type(aten_exc).__name__}: {aten_exc}"
                ),
            )
            return 0

    reference_outputs = reference_value if isinstance(reference_value, tuple) else (reference_value,)
    if aten_name in _CPU_VALIDATION_ONLY_OPS:
        result["ref_mode"] = "cpu_validation_only"
    if is_inplace:
        reference_outputs = (reference_inputs[0],)
    if not all(isinstance(value, torch.Tensor) for value in reference_outputs):
        result.update(status="reference_unsupported", error="reference returned a non-tensor value")
        return 0

    if (
        not is_implicit_inplace
        and len(reference_outputs) != len(out_params)
        and aten_reference is not None
        and aten_reference is not reference
    ):
        try:
            aten_value = aten_reference(*reference_args, **reference_kwargs)
            aten_outputs = aten_value if isinstance(aten_value, tuple) else (aten_value,)
            if len(aten_outputs) == len(out_params):
                reference = aten_reference
                reference_outputs = aten_outputs
                result["ref_mode"] = "aten_functional"
        except Exception:
            pass

    if not is_implicit_inplace and len(reference_outputs) != len(out_params):
        result.update(
            status="reference_unsupported",
            error=(
                f"reference returned {len(reference_outputs)} outputs; "
                f"schema declares {len(out_params)}"
            ),
        )
        return 0

    if aten_name in _COMPARISON_OPS and len(reference_outputs) == 1:
        comparison_out = torch.empty_like(inputs[0])
        try:
            reference(*reference_args, out=comparison_out, **reference_kwargs)
            reference_outputs = (comparison_out,)
            result["ref_mode"] = "out="
        except Exception:
            pass

    unsupported = next(
        (value.dtype for value in reference_outputs if not _supported_infini_dtype(torch, value)),
        None,
    )
    if unsupported is not None:
        result.update(
            status="output_dtype_unsupported",
            error=f"InfiniOps DataType does not support {unsupported}",
        )
        return 0
    infini_inputs = [_clone_value(torch, value) for value in inputs]
    if is_implicit_inplace:
        infini_outputs = (infini_inputs[0],)
        infini_args = infini_inputs
    else:
        output_factory = (
            torch.zeros_like
            if aten_name in _ZERO_INITIALIZED_OUTPUT_OPS
            else torch.empty_like
        )
        infini_outputs = tuple(output_factory(value) for value in reference_outputs)
        infini_args = [*infini_inputs, *infini_outputs]

    infini_function = getattr(infini_ops, schema["name"])

    def call_infini() -> Any:
        return infini_function(
            *infini_args,
            implementation_index=implementation_index,
        )

    try:
        call_infini()
        _synchronize(torch, device)
    except Exception as exc:  # noqa: BLE001
        result.update(
            status="backend_runtime_error",
            error=_clean_error(f"{type(exc).__name__}: {exc}"),
        )
        return 0

    try:
        if aten_name in _NONDETERMINISTIC_OPS:
            result["validation_mode"] = "finite_and_range"
            for actual in infini_outputs:
                if actual.dtype.is_floating_point and not torch.isfinite(actual).all():
                    raise AssertionError("random output contains non-finite values")
                if aten_name == "bernoulli" and not ((actual == 0) | (actual == 1)).all():
                    raise AssertionError("bernoulli output is not binary")
        else:
            for actual, expected in zip(infini_outputs, reference_outputs):
                _assert_close(torch, actual, expected, dtype_name)
    except AssertionError as exc:
        result.update(status="correctness_failed", error=_clean_error(exc))
        return 0

    reference_benchmark_inputs = [_clone_value(torch, value) for value in inputs]
    reference_benchmark_args, reference_benchmark_kwargs = split_reference_arguments(
        reference_params, reference_benchmark_inputs
    )
    reference_benchmark_outputs = tuple(torch.empty_like(value) for value in reference_outputs)

    def call_reference_functional() -> Any:
        return reference(*reference_benchmark_args, **reference_benchmark_kwargs)

    reference_call = call_reference_functional
    if not is_inplace and len(reference_benchmark_outputs) == 1:
        try:
            reference(
                *reference_benchmark_args,
                out=reference_benchmark_outputs[0],
                **reference_benchmark_kwargs,
            )

            def call_reference_out() -> Any:
                return reference(
                    *reference_benchmark_args,
                    out=reference_benchmark_outputs[0],
                    **reference_benchmark_kwargs,
                )

            reference_call = call_reference_out
            result["ref_mode"] = "out="
        except Exception:
            result["ref_mode"] = "functional"

    try:
        infini_us = _measure_us(torch, call_infini, min_run_time, device)
        ref_us = (
            None
            if aten_name in _CPU_VALIDATION_ONLY_OPS
            else _measure_us(torch, reference_call, min_run_time, device)
        )
        _synchronize(torch, device)
    except Exception as exc:  # noqa: BLE001
        result.update(
            status="measure_error",
            error=_clean_error(f"{type(exc).__name__}: {exc}"),
        )
        return 0

    result.update(
        status="success",
        infini_us=round(infini_us, 3),
        ref_us=round(ref_us, 3) if ref_us is not None else None,
        speedup=round(ref_us / infini_us, 4) if ref_us is not None and infini_us else None,
        error="",
    )
    return 0


def _worker_failure_row(
    schema: dict[str, Any],
    dtype: str,
    status: str,
    error: str,
    exit_code: int | None,
    duration_s: float,
) -> dict[str, Any]:
    row = _base_result(schema, dtype)
    row.update(
        status=status,
        error=_clean_error(error),
        worker_exit_code=exit_code,
        duration_s=round(duration_s, 3),
    )
    return row


def run_parent(args: argparse.Namespace) -> int:
    metadata_path = Path(args.metadata).resolve()
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    schemas = canonical_schemas(metadata_payload)
    if args.ops:
        requested = {name.strip() for value in args.ops for name in value.split(",") if name.strip()}
        schemas = [schema for schema in schemas if schema["name"] in requested]
    if args.limit is not None:
        schemas = schemas[: args.limit]

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "results.jsonl"
    prior: dict[str, dict[str, Any]] = {}
    if args.resume and checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("op"):
                prior[row["op"]] = row

    start = utc_now()
    rows = []
    script = str(Path(args.script or sys.argv[0]).resolve())
    position = 0
    batch_size = max(1, args.worker_batch_size)
    for batch_start in range(0, len(schemas), batch_size):
        batch = schemas[batch_start : batch_start + batch_size]
        pending: list[dict[str, Any]] = []
        for schema in batch:
            position += 1
            if schema["name"] in prior:
                row = prior[schema["name"]]
                rows.append(row)
                print(
                    f"[{position}/{len(schemas)}] {schema['name']}: resumed {row['status']}",
                    flush=True,
                )
            else:
                pending.append(schema)

        if not pending:
            continue

        command = build_worker_batch_command(
            python=sys.executable,
            script=script,
            metadata=str(metadata_path),
            op_names=[schema["name"] for schema in pending],
            device=args.device,
            dtype=args.dtype,
            min_run_time=args.min_run_time,
        )
        started = time.monotonic()
        worker_rows: dict[str, dict[str, Any]] = {}
        worker_return_code = None
        worker_stderr = ""
        batch_status = "worker_crash"
        batch_error = ""
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=args.batch_timeout,
                env=os.environ.copy(),
                check=False,
            )
            duration = time.monotonic() - started
            worker_rows = parse_worker_stdout_rows(completed.stdout)
            worker_return_code = completed.returncode
            worker_stderr = completed.stderr or completed.stdout
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - started
            stdout = exc.stdout or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            stderr = exc.stderr or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            worker_rows = parse_worker_stdout_rows(stdout)
            worker_return_code = None
            worker_stderr = stdout + "\n" + stderr
            batch_status = "timeout"
            batch_error = f"worker batch exceeded {args.batch_timeout}s"

        batch_position = position - len(batch)
        for schema in batch:
            batch_position += 1
            if schema["name"] in prior:
                continue
            row = worker_rows.get(schema["name"])
            if row is None:
                row = _worker_failure_row(
                    schema,
                    args.dtype,
                    batch_status,
                    batch_error
                    or (worker_stderr or "worker produced no result record"),
                    worker_return_code,
                    duration,
                )
            else:
                row["worker_exit_code"] = worker_return_code
                row["worker_stderr"] = _clean_error(worker_stderr, 2000)

            rows.append(row)
            with checkpoint_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[{batch_position}/{len(schemas)}] {schema['name']}: {row['status']}",
                flush=True,
            )

    report_metadata = {
        "start": start,
        "end": utc_now(),
        "sha": args.sha,
        "device": args.device,
        "dtype": args.dtype,
        "benchmark_profile": _BENCHMARK_PROFILE,
        "metadata_path": str(metadata_path),
        "metadata_schema_count": len(metadata_payload.get("ops", [])),
        "canonical_wrapper_count": len(canonical_schemas(metadata_payload)),
        "attempted_wrapper_count": len(schemas),
        "per_op_timeout_s": args.timeout,
        "worker_batch_size": batch_size,
        "worker_batch_timeout_s": args.batch_timeout,
        "min_run_time_s": args.min_run_time,
        "command": " ".join(sys.argv),
        "python": sys.version.replace("\n", " "),
    }
    write_reports(output_dir, rows, report_metadata)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark every generated InfiniOps wrapper with fixed inputs."
    )
    parser.add_argument("--metadata", required=True, help="torch_ops_metadata.json")
    parser.add_argument("--output-dir", default="full-operator-results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--sha", default="unknown")
    parser.add_argument("--min-run-time", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--worker-batch-size", type=int, default=1)
    parser.add_argument(
        "--batch-timeout",
        type=float,
        help="wall-clock timeout for one worker batch; defaults to timeout * batch size",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ops", action="append", help="comma-separated wrapper names")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--script", help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-op", help=argparse.SUPPRESS)
    parser.add_argument("--worker-ops", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        if not args.worker_op and not args.worker_ops:
            raise SystemExit("--worker-op or --worker-ops is required in worker mode")
        return run_worker(args)
    if args.batch_timeout is None:
        args.batch_timeout = args.timeout * max(1, args.worker_batch_size)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
