import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_full_operator_report import (
    DIAGNOSTIC_FIELDS,
    PLATFORM_FIELDS,
    main,
    platform_row,
    shape_config,
)


class ExportFullOperatorReportTests(unittest.TestCase):
    def test_shape_config_formats_elementwise_and_matrix_recipes(self):
        self.assertEqual(
            shape_config(
                {
                    "op": "add",
                    "input_recipe": (
                        "input=float16[1x4096]:normal; other=float16[1x4096]:normal"
                    ),
                }
            ),
            "[1, 4096]",
        )
        self.assertEqual(
            shape_config(
                {
                    "op": "addmm",
                    "input_recipe": (
                        "input=float16[1x11008]:normal; mat1=float16[1x4096]:normal; "
                        "mat2=float16[4096x11008]:normal"
                    ),
                }
            ),
            "A=[1, 4096], B=[4096, 11008]",
        )

    def test_platform_row_uses_requested_schema_and_matrix_tflops(self):
        row = platform_row(
            {
                "op": "mm",
                "dtype": "float16",
                "input_recipe": (
                    "self=float16[8x4096]:normal; mat2=float16[4096x11008]:normal"
                ),
                "infini_us": 100.0,
                "ref_us": 80.0,
                "ref_mode": "out=",
                "validation_mode": "exact",
                "implementation_index": 8,
                "active_implementation_indices": [8],
            },
            platform="nvidia",
            run_date="2026/08/30",
            tester="baoming",
        )

        self.assertEqual(list(row), PLATFORM_FIELDS)
        self.assertEqual(row["dtype"], "fp16")
        self.assertEqual(row["category_code"], "linalg")
        self.assertNotEqual(row["ic_tflops"], "NA")
        self.assertIn("implementation=torch-backend", row["remarks"])

    def test_linear_uses_transposed_weight_shape_for_tflops(self):
        row = platform_row(
            {
                "op": "linear",
                "dtype": "float16",
                "input_recipe": (
                    "input=float16[8x4096]:normal; "
                    "weight=float16[11008x4096]:normal"
                ),
                "infini_us": 100.0,
                "ref_us": 80.0,
                "ref_mode": "out=",
                "validation_mode": "allclose",
                "implementation_index": 0,
                "active_implementation_indices": [0],
            },
            platform="ascend",
            run_date="2026/09/07",
            tester="baoming",
        )

        self.assertNotEqual(row["ic_tflops"], "NA")
        self.assertNotEqual(row["pt_tflops"], "NA")

    def test_platform_row_labels_vendor_native_implementation(self):
        for platform in ("cambricon", "ascend", "hygon", "metax"):
            with self.subTest(platform=platform):
                row = platform_row(
                    {
                        "op": "add",
                        "dtype": "float16",
                        "input_recipe": "input=float16[1x4096]:normal",
                        "infini_us": 10.0,
                        "ref_us": 8.0,
                        "ref_mode": "out=",
                        "validation_mode": "exact",
                        "implementation_index": 0,
                        "active_implementation_indices": [0, 8],
                    },
                    platform=platform,
                    run_date="2026/09/06",
                    tester="baoming",
                )

                self.assertIn(f"implementation={platform}-native", row["remarks"])

    def test_platform_row_maps_nine_semantic_category_codes(self):
        expected = {
            "abs": "math",
            "where": "logic",
            "sort": "reduction",
            "matmul": "linalg",
            "softmax": "neural",
            "gather": "indexing",
            "fft_fft": "special",
            "avg_pool2d": "dense",
            "sparse_sampled_addmm": "sparse",
        }
        for op_name, category_code in expected.items():
            with self.subTest(op_name=op_name):
                row = platform_row(
                    {
                        "op": op_name,
                        "dtype": "float16",
                        "input_recipe": "input=float16[1x4096]:normal",
                        "infini_us": 10.0,
                        "ref_us": 8.0,
                        "ref_mode": "functional",
                        "validation_mode": "exact",
                        "implementation_index": 8,
                        "active_implementation_indices": [8],
                    },
                    platform="metax",
                    run_date="2026/09/15",
                    tester="baoming",
                )
                self.assertEqual(row["category_code"], category_code)

    def test_main_separates_successes_and_diagnostics_and_deduplicates(self):
        success = {
            "op": "abs",
            "dtype": "float16",
            "input_recipe": "input=float16[1x4096]:normal",
            "status": "success",
            "infini_us": 10.0,
            "ref_us": 5.0,
            "ref_mode": "out=",
            "validation_mode": "exact",
            "implementation_index": 8,
            "active_implementation_indices": [8],
        }
        failure = {
            "op": "svd",
            "dtype": "float16",
            "input_recipe": "input=float16[32x32]:normal",
            "status": "correctness_failed",
            "error": "mismatch",
            "benchmark_profile": "decode-4k",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "results.json"
            source.write_text(
                json.dumps({"results": [success, dict(success), failure]}), encoding="utf-8"
            )
            output = root / "benchmark.csv"
            diagnostics = root / "diagnostics.csv"
            report = root / "report.md"

            exit_code = main(
                [
                    str(source),
                    "--output",
                    str(output),
                    "--diagnostics",
                    str(diagnostics),
                    "--report",
                    str(report),
                ]
            )

            self.assertEqual(exit_code, 0)
            with output.open(encoding="utf-8", newline="") as handle:
                success_rows = list(csv.DictReader(handle))
            with diagnostics.open(encoding="utf-8", newline="") as handle:
                diagnostic_rows = list(csv.DictReader(handle))
                self.assertEqual(handle.seek(0), 0)
                self.assertEqual(next(csv.reader(handle)), DIAGNOSTIC_FIELDS)
            self.assertEqual(len(success_rows), 1)
            self.assertEqual(success_rows[0]["category_code"], "math")
            self.assertEqual(len(diagnostic_rows), 1)
            self.assertIn("Successful comparable cases: 1", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
