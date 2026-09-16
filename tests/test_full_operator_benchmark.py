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
    _load_device_plugin,
    _manual_seed_all,
    _resolve_aten_reference,
    _synchronize,
    build_worker_command,
    build_worker_batch_command,
    canonical_schemas,
    parse_worker_stdout,
    parse_worker_stdout_rows,
    recipe_for_tensor,
    recipe_for_tensor_list,
    scalar_value,
    schema_parameter_groups,
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

        self.assertEqual(command[-2:], ["--min-run-time", "0.05"])
        start = command.index("--worker-op")
        self.assertEqual(command[start : start + 2], ["--worker-op", "abs"])

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
