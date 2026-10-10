import csv
import importlib
import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.full_operator_benchmark import (
    _device_type,
    _make_tensor,
    _load_device_plugin,
    _manual_seed_all,
    _measure_us,
    _reference_function,
    _resolve_aten_reference,
    _relate_inputs,
    _synchronize,
    _supports_infini_only_reference_fallback,
    _try_infini_only_reference_fallback,
    run_parent,
    select_dtype_supported_schemas,
    select_platform_supported_schemas,
    build_worker_command,
    build_worker_batch_command,
    build_parser,
    canonical_schemas,
    parse_worker_stdout,
    parse_worker_stdout_rows,
    recipe_for_tensor,
    recipe_for_tensor_list,
    scalar_value,
    schema_parameter_groups,
    select_schemas,
    split_reference_arguments,
    summarize_results,
    write_reports,
)
from scripts import full_operator_benchmark as _harness_module


class FullOperatorBenchmarkTests(unittest.TestCase):
    def test_mlu_runtime_helpers_load_plugin_seed_and_synchronize(self):
        accelerator = Mock()
        torch = Mock(mlu=accelerator)

        self.assertEqual(_device_type("mlu:1"), "mlu")
        with patch("scripts.full_operator_benchmark.importlib.import_module") as load:
            _load_device_plugin("mlu:1")
        load.assert_called_once_with("torch_mlu")

        _manual_seed_all(torch, "mlu:1", 20260905)
        accelerator.manual_seed_all.assert_called_once_with(20260905)

        _synchronize(torch, "mlu:1")
        accelerator.synchronize.assert_called_once_with()

    def test_npu_runtime_helpers_load_plugin_seed_and_synchronize(self):
        accelerator = Mock()
        torch = Mock(npu=accelerator)

        self.assertEqual(_device_type("npu:0"), "npu")
        with patch("scripts.full_operator_benchmark.importlib.import_module") as load:
            _load_device_plugin("npu:0")
        load.assert_called_once_with("torch_npu")

        _manual_seed_all(torch, "npu:0", 20260906)
        accelerator.manual_seed_all.assert_called_once_with(20260906)

        _synchronize(torch, "npu:0")
        accelerator.synchronize.assert_called_once_with()

    def test_schema_parameter_groups_preserve_explicit_out_inplace_wrappers(self):
        schema = {
            "name": "add",
            "aten_name": "add_",
            "params": [
                {"name": "input", "is_out": False},
                {"name": "other", "is_out": False},
                {"name": "alpha", "is_out": False},
                {"name": "out", "is_out": True},
            ],
        }

        is_inplace, is_implicit, inputs, outputs = schema_parameter_groups(schema)

        self.assertTrue(is_inplace)
        self.assertFalse(is_implicit)
        self.assertEqual([param["name"] for param in inputs], ["input", "other", "alpha"])
        self.assertEqual([param["name"] for param in outputs], ["out"])

    def test_canonical_schemas_keep_first_wrapper_schema(self):
        metadata = {
            "ops": [
                {"name": "std", "aten_name": "std", "overload_name": "std.dim"},
                {
                    "name": "std",
                    "aten_name": "std",
                    "overload_name": "std.correction",
                },
                {"name": "abs", "aten_name": "abs", "overload_name": "abs"},
            ]
        }

        schemas = canonical_schemas(metadata)

        self.assertEqual([schema["name"] for schema in schemas], ["abs", "std"])
        self.assertEqual(schemas[1]["overload_name"], "std.dim")

    def test_special_category_is_excluded_even_for_explicit_profiles(self):
        schemas = [
            {"name": "abs"},
            {"name": "fft_fft"},
            {"name": "special_gammaln"},
        ]

        selected, excluded = select_schemas(
            schemas,
            requested_ops={"abs", "fft_fft", "special_gammaln"},
            excluded_categories={"special"},
        )

        self.assertEqual([schema["name"] for schema in selected], ["abs"])
        self.assertEqual(
            [schema["name"] for schema in excluded],
            ["fft_fft", "special_gammaln"],
        )

    def test_platform_support_matrix_splits_backend_and_fallback_gaps(self):
        schemas = [
            {"name": "abs"},
            {"name": "histogram"},
            {"name": "igamma"},
        ]

        selected, backend_excluded, fallback_excluded = (
            select_platform_supported_schemas(
                schemas,
                platform="moore-s5000",
                dtype="float16",
            )
        )

        self.assertEqual([schema["name"] for schema in selected], ["abs"])
        self.assertEqual(
            [schema["name"] for schema in backend_excluded],
            ["histogram"],
        )
        self.assertEqual(
            [schema["name"] for schema in fallback_excluded],
            ["igamma"],
        )

        selected, backend_excluded, fallback_excluded = (
            select_platform_supported_schemas(
                [
                    {"name": "batch_norm_elemt"},
                    {"name": "special_legendre_polynomial_p"},
                ],
                platform="moore-s5000",
                dtype="float32",
            )
        )
        self.assertEqual(selected, [])
        self.assertEqual(
            [schema["name"] for schema in backend_excluded],
            ["batch_norm_elemt"],
        )
        self.assertEqual(
            [schema["name"] for schema in fallback_excluded],
            ["special_legendre_polynomial_p"],
        )

    def test_loss_benchmarks_use_elementwise_reduction_and_profile_sized_ranges(self):
        for op_name in (
            "binary_cross_entropy",
            "huber_loss",
            "mse_loss",
            "smooth_l1_loss",
            "soft_margin_loss",
        ):
            self.assertEqual(
                scalar_value(op_name, {"name": "reduction", "type": "int"}),
                0,
            )

        with patch.object(_harness_module, "_DEFAULT_SHAPE", [1, 4096]):
            self.assertEqual(
                scalar_value("linspace", {"name": "steps", "type": "int"}),
                4096,
            )
            self.assertEqual(
                scalar_value("logspace", {"name": "steps", "type": "int"}),
                4096,
            )
            self.assertEqual(
                scalar_value("logspace", {"name": "base", "type": "float"}),
                10.0,
            )

    def test_loss_and_range_references_use_metadata_compatible_overloads(self):
        class FakeAten:
            binary_cross_entropy = Mock()
            huber_loss = Mock()
            linspace = Mock()
            logspace = Mock()

        class FakeOps:
            aten = FakeAten()

        class FakeTorch:
            ops = FakeOps()

        torch = FakeTorch()
        self.assertIs(_reference_function(torch, "binary_cross_entropy"), FakeAten.binary_cross_entropy.default)
        self.assertIs(_reference_function(torch, "huber_loss"), FakeAten.huber_loss.default)
        self.assertIs(_reference_function(torch, "linspace"), FakeAten.linspace.Tensor_Scalar)
        self.assertIs(_reference_function(torch, "logspace"), FakeAten.logspace.Tensor_Scalar)
        self.assertIn("linspace", _harness_module._CPU_VALIDATION_ONLY_OPS)
        self.assertIn("logspace", _harness_module._CPU_VALIDATION_ONLY_OPS)

    def test_range_end_is_converted_to_python_scalar_for_tensor_start_overload(self):
        class FakeTensor:
            device = "cuda"

            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        related, note = _relate_inputs(
            Mock(),
            "linspace",
            [{"name": "start"}, {"name": "end"}, {"name": "steps"}],
            [FakeTensor(-1.0), FakeTensor(1.0), 4096],
            "cuda",
            "float16",
        )

        self.assertEqual(related[1], 1.0)
        self.assertIn("end converted", note)

    def test_masked_select_uses_an_all_true_mask_for_stable_output_storage(self):
        with patch.object(_harness_module, "_DEFAULT_SHAPE", [1, 4096]):
            recipe = recipe_for_tensor(
                "masked_select",
                "mask",
                tensor_index=1,
                default_dtype="float16",
            )

        self.assertEqual(recipe["kind"], "all_true")
        self.assertEqual(recipe["dtype"], "bool")

        class FakeTensor:
            def __init__(self, shape, dtype, device):
                self.shape = shape
                self.dtype = dtype
                self.device = device

        class FakeTorch:
            bool = "bool"

            @staticmethod
            def ones(shape, dtype, device):
                return FakeTensor(shape, dtype, device)

        mask = _make_tensor(FakeTorch(), recipe, "cuda")
        self.assertEqual(mask.shape, [1, 4096])
        self.assertEqual(mask.dtype, "bool")

    def test_category_exclusion_defaults_to_none_and_supports_special(self):
        self.assertEqual(
            build_parser().parse_args(["--metadata", "x"]).exclude_categories,
            "none",
        )
        args = build_parser().parse_args(
            ["--metadata", "x", "--exclude-categories", "special"]
        )
        self.assertEqual(args.exclude_categories, "special")

    def test_fp32_only_special_ops_are_skipped_for_low_precision_dtypes(self):
        schemas = [
            {"name": "abs"},
            {"name": "fft_fft"},
            {"name": "fft_hfft"},
            {"name": "linalg_svd"},
            {"name": "special_zeta"},
        ]

        for dtype in ("float16", "bfloat16"):
            with self.subTest(dtype=dtype):
                selected, excluded = select_dtype_supported_schemas(
                    schemas,
                    dtype=dtype,
                )
                self.assertEqual([schema["name"] for schema in selected], ["abs"])
                self.assertEqual(
                    [schema["name"] for schema in excluded],
                    ["fft_fft", "fft_hfft", "linalg_svd", "special_zeta"],
                )

        selected, excluded = select_dtype_supported_schemas(
            schemas,
            dtype="float32",
        )
        self.assertEqual(len(selected), 5)
        self.assertEqual(excluded, [])
        self.assertEqual(len(_harness_module._FP32_ONLY_SPECIAL_OPS), 53)
        self.assertEqual(len(_harness_module._FP32_ONLY_LOW_PRECISION_OPS), 69)
        self.assertEqual(len(_harness_module._FP32_ONLY_OPS), 122)

    def test_low_precision_dtype_exclusions_are_recorded_in_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "torch_ops_metadata.json"
            output = root / "output"
            metadata.write_text(
                json.dumps({"ops": [{"name": "special_zeta"}]}),
                encoding="utf-8",
            )

            args = build_parser().parse_args(
                [
                    "--metadata",
                    str(metadata),
                    "--output-dir",
                    str(output),
                    "--dtype",
                    "float16",
                ]
            )
            self.assertEqual(run_parent(args), 0)
            payload = json.loads(
                (output / "excluded-operators.json").read_text(encoding="utf-8")
            )

        self.assertEqual(payload["dtype"], "float16")
        self.assertEqual(payload["dtype_excluded_count"], 1)
        self.assertEqual(
            payload["dtype_excluded_operators"],
            ["special_zeta"],
        )

    def test_reference_failure_falls_back_to_infini_only_special_function(self):
        class FakeTensor:
            dtype = "bfloat16"
            shape = [2, 4]

            def clone(self):
                return self

        class FakeTorch:
            Tensor = FakeTensor
            bfloat16 = "bfloat16"
            float16 = "float16"
            float32 = "float32"
            float64 = "float64"

            @staticmethod
            def broadcast_shapes(*shapes):
                return list(shapes[0])

            @staticmethod
            def empty(shape, dtype, device):
                return FakeTensor()

        class FakeOps:
            def special_bessel_y1(self, *args, implementation_index):
                self.args = args
                self.implementation_index = implementation_index

        schema = {
            "name": "special_bessel_y1",
            "aten_name": "special_bessel_y1",
            "params": [
                {"name": "x", "is_tensor": True, "is_out": False},
                {"name": "out", "is_tensor": True, "is_out": True},
            ],
        }
        result = {
            "status": "worker_error",
            "infini_us": None,
            "ref_us": None,
            "speedup": None,
            "ref_mode": "functional",
            "validation_mode": "exact",
            "reference_error": "",
            "error": "",
        }
        fake_ops = FakeOps()

        with patch.object(_harness_module, "_measure_us", return_value=12.5):
            handled = _try_infini_only_reference_fallback(
                torch=FakeTorch(),
                infini_ops=fake_ops,
                schema=schema,
                result=result,
                inputs=[FakeTensor()],
                device="cpu",
                min_run_time=0.02,
                benchmark_iterations=1,
                implementation_index=3,
                reference_error="RuntimeError: dtype is unsupported",
            )

        self.assertTrue(handled)
        self.assertTrue(_supports_infini_only_reference_fallback(schema))
        self.assertFalse(
            _supports_infini_only_reference_fallback(
                {**schema, "name": "fft_fft", "aten_name": "fft_fft"}
            )
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["infini_us"], 12.5)
        self.assertIsNone(result["ref_us"])
        self.assertEqual(result["ref_mode"], "unavailable")
        self.assertEqual(result["validation_mode"], "infini_only")
        self.assertIn("dtype is unsupported", result["reference_error"])
        self.assertEqual(fake_ops.implementation_index, 3)

    def test_fixed_tensor_recipes_cover_matrix_index_and_domains(self):
        self.assertEqual(
            recipe_for_tensor("mm", "self", 0),
            {"shape": [64, 128], "dtype": "float16", "kind": "normal"},
        )
        self.assertEqual(recipe_for_tensor("mm", "mat2", 1)["shape"], [128, 64])

        index_recipe = recipe_for_tensor("gather", "index", 1)
        self.assertEqual(index_recipe["dtype"], "int64")
        self.assertEqual(index_recipe["kind"], "bounded_index")
        self.assertEqual(index_recipe["high"], 128)

        self.assertEqual(recipe_for_tensor("sqrt", "input", 0)["kind"], "positive")
        self.assertEqual(recipe_for_tensor("acos", "input", 0)["kind"], "unit_interval")
        self.assertEqual(
            recipe_for_tensor("binary_cross_entropy", "input", 0)["kind"],
            "probability",
        )
        self.assertEqual(
            recipe_for_tensor("binary_cross_entropy", "target", 1)["dtype"],
            "float16",
        )
        self.assertEqual(recipe_for_tensor("nll_loss", "target", 1)["dtype"], "int64")
        self.assertEqual(
            recipe_for_tensor("avg_pool2d", "input", 0)["shape"],
            [2, 8, 16, 16],
        )

    def test_diff_elu_backward_and_fractional_pool_scalar_contracts(self):
        self.assertEqual(scalar_value("diff", {"name": "n", "type": "int"}), 1)
        self.assertEqual(scalar_value("diff", {"name": "dim", "type": "int"}), -1)
        self.assertEqual(
            scalar_value("elu_backward", {"name": "alpha", "type": "Scalar"}),
            1.0,
        )
        self.assertEqual(
            scalar_value(
                "fractional_max_pool2d",
                {"name": "kernel_size", "type": "int[2]"},
            ),
            [2, 2],
        )
        self.assertEqual(
            recipe_for_tensor("fractional_max_pool2d", "random_samples", 1),
            {"shape": [2, 8, 2], "dtype": "float16", "kind": "probability"},
        )
        self.assertEqual(
            recipe_for_tensor("fractional_max_pool3d", "random_samples", 1)["shape"],
            [2, 4, 3],
        )

    def test_sparse_gemm_and_pinv_use_compatible_tensor_shapes(self):
        self.assertEqual(
            [recipe_for_tensor("sspaddmm", name, index)["shape"] for index, name in enumerate(("input", "mat1", "mat2"))],
            [[8, 16], [8, 128], [128, 16]],
        )
        self.assertEqual(
            recipe_for_tensor("linalg_pinv", "input", 0)["kind"],
            "positive_definite",
        )

    def test_semantic_tensor_recipes_cover_condition_rank_and_square_matrices(self):
        self.assertEqual(
            recipe_for_tensor("where", "condition", 0),
            {"shape": [8, 128], "dtype": "bool", "kind": "boolean"},
        )
        self.assertEqual(
            recipe_for_tensor("upsample_linear1d", "input", 0)["shape"],
            [2, 4, 16],
        )
        self.assertEqual(
            recipe_for_tensor("upsample_bilinear2d", "input", 0)["shape"],
            [2, 4, 16, 16],
        )
        self.assertEqual(
            recipe_for_tensor("upsample_trilinear3d", "input", 0)["shape"],
            [2, 2, 8, 8, 8],
        )
        self.assertEqual(
            recipe_for_tensor("triangular_solve", "A", 1)["shape"],
            [32, 32],
        )
        self.assertEqual(
            recipe_for_tensor("cholesky", "input", 0)["kind"],
            "positive_definite",
        )
        self.assertEqual(
            recipe_for_tensor("scatter", "index", 1)["kind"],
            "unique_index",
        )
        self.assertEqual(
            recipe_for_tensor("linalg_det", "A", 0)["shape"],
            [32, 32],
        )
        self.assertEqual(
            recipe_for_tensor("linalg_cross", "other", 1)["shape"],
            [4096, 3],
        )
        self.assertEqual(
            recipe_for_tensor("bucketize", "boundaries", 1)["shape"],
            [64],
        )

    def test_operator_specific_recipes_cover_linked_input_families(self):
        self.assertEqual(
            recipe_for_tensor("reflection_pad2d_backward", "input", 1)["shape"],
            [2, 4, 16, 16],
        )
        self.assertEqual(
            recipe_for_tensor("reflection_pad2d_backward", "grad_output", 0)[
                "shape"
            ],
            [2, 4, 18, 18],
        )
        self.assertEqual(
            recipe_for_tensor("_slow_conv2d_forward", "weight", 1)["shape"],
            [16, 8, 3, 3],
        )
        self.assertEqual(
            recipe_for_tensor("slow_conv3d_forward", "weight", 1)["shape"],
            [16, 8, 3, 3, 3],
        )
        self.assertEqual(
            recipe_for_tensor("histogram", "bins", 1)["shape"],
            [65],
        )
        self.assertEqual(
            recipe_for_tensor("nll_loss2d_forward", "target", 1),
            {
                "shape": [2, 8, 8],
                "dtype": "int64",
                "kind": "bounded_index",
                "high": 16,
            },
        )
        self.assertEqual(
            recipe_for_tensor("linspace", "start", 0)["shape"],
            [],
        )
        self.assertEqual(
            recipe_for_tensor("linalg_tensorinv", "input", 0)["shape"],
            [4, 4, 4, 4],
        )
        self.assertEqual(
            recipe_for_tensor("_int_mm", "input", 0)["dtype"],
            "int8",
        )

    def test_tensor_list_recipes_are_shape_compatible(self):
        cat = recipe_for_tensor_list("cat", "tensors")
        self.assertEqual(cat["kind"], "tensor_list")
        self.assertEqual(cat["shapes"], [[8, 128], [8, 128]])

        chain = recipe_for_tensor_list("chain_matmul", "matrices")
        self.assertEqual(chain["shapes"], [[32, 48], [48, 24], [24, 16]])

    def test_scalar_values_use_safe_operator_specific_defaults(self):
        self.assertEqual(scalar_value("softmax", {"name": "dim", "type": "int"}), -1)
        self.assertEqual(scalar_value("kthvalue", {"name": "k", "type": "int"}), 1)
        self.assertEqual(
            scalar_value("avg_pool2d", {"name": "kernel_size", "type": "int[2]"}),
            [2, 2],
        )
        self.assertEqual(
            scalar_value("avg_pool2d", {"name": "stride", "type": "int[2]"}),
            [2, 2],
        )
        self.assertEqual(
            scalar_value("hardtanh", {"name": "min_val", "type": "Scalar"}),
            -1.0,
        )
        self.assertEqual(
            scalar_value("amax", {"name": "dim", "type": "int[]"}),
            [-1],
        )
        self.assertEqual(
            scalar_value("multinomial", {"name": "num_samples", "type": "SymInt"}),
            4,
        )
        self.assertEqual(
            scalar_value("linalg_qr", {"name": "mode", "type": "str"}),
            "reduced",
        )
        self.assertEqual(
            scalar_value("softplus_backward", {"name": "threshold", "type": "Scalar"}),
            20.0,
        )
        self.assertEqual(
            scalar_value(
                "reflection_pad3d",
                {"name": "padding", "type": "SymInt[6]"},
            ),
            [1, 1, 1, 1, 1, 1],
        )
        self.assertEqual(
            scalar_value("quantile", {"name": "interpolation", "type": "str"}),
            "linear",
        )
        self.assertEqual(
            scalar_value("special_multigammaln", {"name": "p", "type": "int"}),
            2,
        )
        self.assertTrue(
            scalar_value("linalg_lu_solve", {"name": "left", "type": "bool"})
        )

    def test_decode_4k_small_profile_uses_small_default_and_matrix_shapes(self):
        with patch.dict(
            os.environ,
            {"INFINIOPS_BENCHMARK_PROFILE": "decode-4k-small"},
            clear=True,
        ):
            module = importlib.reload(_harness_module)

            try:
                self.assertEqual(module._DEFAULT_SHAPE, [1, 128])
                self.assertEqual(
                    module._MATRIX_SHAPES["matmul"],
                    [[1, 128], [128, 256]],
                )
            finally:
                os.environ.pop("INFINIOPS_BENCHMARK_PROFILE", None)
                importlib.reload(_harness_module)

    def test_worker_command_is_reproducible(self):
        command = build_worker_command(
            python="/usr/bin/python3.12",
            script="/workspace/benchmark_infiniops_full.py",
            metadata="/workspace/torch_ops_metadata.json",
            op_name="abs",
            device="cuda",
            dtype="float16",
            min_run_time=0.05,
        )

        min_run_index = command.index("--min-run-time")
        self.assertEqual(command[min_run_index:min_run_index + 2], ["--min-run-time", "0.05"])
        self.assertIn("--benchmark-iterations", command)
        iterations_index = command.index("--benchmark-iterations")
        self.assertEqual(
            command[iterations_index:iterations_index + 2],
            ["--benchmark-iterations", "0"],
        )
        start = command.index("--worker-op")
        self.assertEqual(command[start : start + 2], ["--worker-op", "abs"])

    def test_worker_command_can_request_fixed_iterations(self):
        command = build_worker_command(
            python="/usr/bin/python3.12",
            script="/workspace/benchmark_infiniops_full.py",
            metadata="/workspace/torch_ops_metadata.json",
            op_name="abs",
            device="cuda",
            dtype="float16",
            min_run_time=0.02,
            benchmark_iterations=1,
        )

        iterations_index = command.index("--benchmark-iterations")
        self.assertEqual(
            command[iterations_index:iterations_index + 2],
            ["--benchmark-iterations", "1"],
        )

    def test_worker_batch_command_groups_ops_in_one_process(self):
        command = build_worker_batch_command(
            python="/usr/bin/python3.12",
            script="/workspace/benchmark_infiniops_full.py",
            metadata="/workspace/torch_ops_metadata.json",
            op_names=["abs", "add"],
            device="cuda",
            dtype="float16",
            min_run_time=0.05,
        )

        start = command.index("--worker-ops")
        self.assertEqual(command[start : start + 2], ["--worker-ops", "abs,add"])
        self.assertNotIn("--worker-op", command)
        self.assertIn("--benchmark-iterations", command)

    def test_fixed_iterations_use_benchmark_timer_timeit(self):
        class Measurement:
            def __init__(self, raw_seconds, number_per_run=1):
                self.raw_times = [raw_seconds]
                self.number_per_run = number_per_run
                self.median = raw_seconds / number_per_run

        class Timer:
            calls = []

            def __init__(self, stmt, globals):
                self.stmt = stmt
                self.globals = globals

            def timeit(self, number):
                Timer.calls.append(("timeit", {"number": number}))
                return Measurement(0.000012, number)

            def blocked_autorange(self, min_run_time):
                Timer.calls.append(
                    ("blocked_autorange", {"min_run_time": min_run_time})
                )
                return Measurement(0.000012)

        class Benchmark:
            pass

        Benchmark.Timer = Timer

        torch = Mock()
        torch.utils.benchmark = Benchmark()
        latency_us = _measure_us(
            torch, lambda: None, 0.02, "cuda", benchmark_iterations=1
        )

        self.assertEqual(latency_us, 12.0)
        self.assertEqual(
            Timer.calls,
            [("timeit", {"number": 1})],
        )

    def test_reference_keyword_only_parameters_are_not_passed_positionally(self):
        params = [
            {"name": "input"},
            {"name": "dim", "keyword_only": False},
            {"name": "index"},
            {"name": "sparse_grad", "keyword_only": True},
        ]
        values = ["input", -1, "index", False]

        positional, keyword = split_reference_arguments(params, values)

        self.assertEqual(positional, ["input", -1, "index"])
        self.assertEqual(keyword, {"sparse_grad": False})

        addmm_params = [
            {"name": "input"},
            {"name": "mat1"},
            {"name": "mat2"},
            {"name": "beta", "keyword_only": True},
            {"name": "alpha", "keyword_only": True},
        ]
        positional, keyword = split_reference_arguments(
            addmm_params, ["self", "a", "b", 1.0, 1.0]
        )
        self.assertEqual(positional, ["self", "a", "b"])
        self.assertEqual(keyword, {"beta": 1.0, "alpha": 1.0})

        gelu_params = [
            {"name": "input"},
            {"name": "approximate", "keyword_only": True},
        ]
        positional, keyword = split_reference_arguments(gelu_params, ["input", "none"])
        self.assertEqual(positional, ["input"])
        self.assertEqual(keyword, {"approximate": "none"})

    def test_aten_schema_drives_keyword_only_layout(self):
        class Argument:
            def __init__(self, name, aten_type, keyword_only):
                self.name = name
                self.type = aten_type
                self.kwarg_only = keyword_only

        class Schema:
            arguments = [
                Argument("self", "Tensor", False),
                Argument("dim", "int", False),
                Argument("index", "Tensor", False),
                Argument("sparse_grad", "bool", True),
            ]

        class Overload:
            _schema = Schema()

        class Packet:
            default = Overload()

            @staticmethod
            def overloads():
                return ["default"]

        class Aten:
            gather = Packet()

        class Ops:
            aten = Aten()

        class Torch:
            ops = Ops()

        params = [
            {"name": "input", "type": "Tensor"},
            {"name": "dim", "type": "int"},
            {"name": "index", "type": "Tensor"},
            {"name": "sparse_grad", "type": "bool"},
        ]

        overload, annotated = _resolve_aten_reference(Torch(), "gather", params)

        self.assertIs(overload, Packet.default)
        self.assertEqual(
            [param["keyword_only"] for param in annotated],
            [False, False, False, True],
        )

    def test_aten_reference_arguments_are_reordered_by_parameter_name(self):
        class Argument:
            def __init__(self, name, aten_type, keyword_only=False):
                self.name = name
                self.type = aten_type
                self.kwarg_only = keyword_only

        class Schema:
            arguments = [
                Argument("grad_output", "Tensor"),
                Argument("alpha", "Scalar"),
                Argument("self_or_result", "Tensor"),
            ]

        class Overload:
            _schema = Schema()

        class Packet:
            default = Overload()

            @staticmethod
            def overloads():
                return ["default"]

        class Torch:
            class ops:
                class aten:
                    elu_backward = Packet()

        params = [
            {"name": "grad_output", "type": "Tensor"},
            {"name": "self_or_result", "type": "Tensor"},
            {"name": "alpha", "type": "Scalar"},
        ]

        _, reference_params = _resolve_aten_reference(
            Torch(),
            "elu_backward",
            params,
        )
        values_by_name = {
            "grad_output": "grad",
            "self_or_result": "self",
            "alpha": 1.0,
        }
        reference_inputs = [
            values_by_name[param["name"]] for param in reference_params
        ]
        positional, keyword = split_reference_arguments(
            reference_params,
            reference_inputs,
        )

        self.assertEqual(
            [param["name"] for param in reference_params],
            ["grad_output", "alpha", "self_or_result"],
        )
        self.assertEqual(positional, ["grad", 1.0, "self"])
        self.assertEqual(keyword, {})

    def test_aten_reference_uses_selected_overload_not_last_candidate(self):
        class Argument:
            def __init__(self, name, aten_type, keyword_only=False):
                self.name = name
                self.type = aten_type
                self.kwarg_only = keyword_only

        class Schema:
            def __init__(self, arguments):
                self.arguments = arguments

        class Overload:
            def __init__(self, arguments):
                self._schema = Schema(arguments)

        class Packet:
            src = Overload(
                [
                    Argument("self", "Tensor"),
                    Argument("dim", "int"),
                    Argument("index", "Tensor"),
                    Argument("src", "Tensor"),
                ]
            )
            optional = Overload(
                [
                    Argument("self", "Tensor"),
                    Argument("dim", "int"),
                    Argument("index", "Tensor"),
                    Argument("generator", "Optional[Generator]"),
                ]
            )

            @staticmethod
            def overloads():
                return ["src", "optional"]

        class Torch:
            class ops:
                class aten:
                    scatter = Packet()

        params = [
            {"name": "input", "type": "Tensor"},
            {"name": "index", "type": "Tensor"},
            {"name": "src", "type": "Tensor"},
            {"name": "dim", "type": "int"},
        ]

        overload, reference_params = _resolve_aten_reference(
            Torch(),
            "scatter",
            params,
        )

        self.assertIs(overload, Packet.src)
        self.assertEqual(
            [param["name"] for param in reference_params],
            ["input", "dim", "index", "src"],
        )

    def test_index_family_inputs_use_a_shared_dimension_and_bounds(self):
        class FakeTensor:
            dtype = "float32"
            device = "cuda"

            def __init__(self, shape, tag=None):
                self.shape = shape
                self.tag = tag

        class FakeTorch:
            int64 = "int64"

            @staticmethod
            def randint(low, high, shape, dtype, device):
                return FakeTensor(shape, f"index:{low}:{high}:{dtype}")

            @staticmethod
            def randn(shape, dtype, device):
                return FakeTensor(shape, f"source:{dtype}")

        params = [
            {"name": "input"},
            {"name": "index"},
            {"name": "source"},
            {"name": "dim"},
        ]
        inputs = [FakeTensor([8, 128]), "bad-index", "bad-source", "bad-dim"]

        related, note = _relate_inputs(
            FakeTorch(),
            "index_add",
            params,
            inputs,
            "cuda",
            "float32",
        )

        self.assertEqual(related[0].shape, [8, 128])
        self.assertEqual(related[1].shape, [4])
        self.assertEqual(related[1].tag, "index:0:128:int64")
        self.assertEqual(related[2].shape, [8, 4])
        self.assertEqual(related[3], 1)
        self.assertIn("dim=1", note)

    def test_parse_worker_stdout_uses_last_result_record(self):
        stdout = "warning from vendor\n" + json.dumps(
            {"record_type": "result", "op": "abs", "status": "success"}
        )

        self.assertEqual(parse_worker_stdout(stdout)["op"], "abs")
        self.assertIsNone(parse_worker_stdout("only noise"))

    def test_parse_worker_stdout_rows_keeps_each_result(self):
        stdout = "\n".join(
            [
                json.dumps({"record_type": "result", "op": "abs", "status": "success"}),
                "vendor warning",
                json.dumps({"record_type": "result", "op": "add", "status": "timeout"}),
            ]
        )

        rows = parse_worker_stdout_rows(stdout)

        self.assertEqual(sorted(rows), ["abs", "add"])
        self.assertEqual(rows["add"]["status"], "timeout")

    def test_summary_and_reports_include_failures(self):
        rows = [
            {
                "op": "abs",
                "aten_name": "abs",
                "status": "success",
                "infini_us": 12.5,
                "ref_us": 10.0,
                "dtype": "float16",
                "input_recipe": "input=float16[8,128]:normal",
                "error": "",
            },
            {
                "op": "svd",
                "aten_name": "svd",
                "status": "recipe_unsupported",
                "infini_us": None,
                "ref_us": None,
                "dtype": "float16",
                "input_recipe": "",
                "error": "requires a conditioned matrix",
            },
        ]

        summary = summarize_results(rows)
        self.assertEqual(summary["attempted"], 2)
        self.assertEqual(summary["success"], 1)
        self.assertEqual(summary["by_status"]["recipe_unsupported"], 1)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_reports(
                output,
                rows,
                metadata={"sha": "deadbeef", "device": "cuda", "command": "bench"},
            )

            payload = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["attempted"], 2)

            with (output / "results.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                csv_rows = list(csv.DictReader(handle))

            self.assertEqual(len(csv_rows), 2)
            self.assertEqual(csv_rows[1]["status"], "recipe_unsupported")
            self.assertIn(
                "recipe_unsupported",
                (output / "summary.md").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
