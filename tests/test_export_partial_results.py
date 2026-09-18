import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_full_operator_report import _load_rows
from scripts.export_partial_results import main


class ExportPartialResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.success = {
            "record_type": "result",
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
            "benchmark_profile": "decode-4k",
        }
        cls.failure = {
            "record_type": "result",
            "op": "svd",
            "dtype": "float16",
            "input_recipe": "input=float16[32x32]:normal",
            "status": "correctness_failed",
            "error": "mismatch",
            "benchmark_profile": "decode-4k",
        }

    def test_load_rows_accepts_checkpoint_jsonl_and_ignores_truncated_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "results.jsonl"
            checkpoint.write_text(
                json.dumps(self.success) + "\n" + '{"record_type": "result", "op":',
                encoding="utf-8",
            )
            self.assertEqual(_load_rows([checkpoint]), [self.success])

    def test_load_rows_skips_truncated_results_json_but_reads_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_json = root / "results.json"
            checkpoint = root / "results.jsonl"
            result_json.write_text('{"results": [{"record_type"', encoding="utf-8")
            checkpoint.write_text(json.dumps(self.success) + "\n", encoding="utf-8")
            self.assertEqual(_load_rows([result_json, checkpoint]), [self.success])

    def test_partial_export_reads_complete_and_checkpoint_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            section = root / "float16" / "decode-4k"
            section.mkdir(parents=True)
            (section / "results.json").write_text(
                json.dumps({"results": [self.success, dict(self.success)]}),
                encoding="utf-8",
            )
            interrupted = root / "float32" / "decode-4k"
            interrupted.mkdir(parents=True)
            (interrupted / "results.jsonl").write_text(
                json.dumps(self.failure) + "\n", encoding="utf-8"
            )

            exit_code = main(
                [
                    "--output-root",
                    str(root),
                    "--output-name",
                    "infiniops-test-device.csv",
                    "--platform",
                    "test",
                    "--date",
                    "2026/09/18",
                    "--infiniops-commit",
                    "1111111111111111111111111111111111111111",
                    "--infinirt-commit",
                    "2222222222222222222222222222222222222222",
                    "--failure-category",
                    "benchmark_failure",
                    "--failure-reason",
                    "interrupted for test",
                ]
            )

            self.assertEqual(exit_code, 0)
            partial_csv = root / "infiniops-test-device.partial.csv"
            diagnostics = root / "diagnostics.partial.csv"
            manifest_path = root / "partial-manifest.json"
            self.assertTrue(partial_csv.exists())
            self.assertTrue(diagnostics.exists())
            self.assertTrue((root / "report.partial.md").exists())

            with partial_csv.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["op_name"], "abs")
            self.assertEqual(
                rows[0]["infiniops_commit"], "1111111111111111111111111111111111111111"
            )

            with diagnostics.open(encoding="utf-8", newline="") as handle:
                diagnostic_rows = list(csv.DictReader(handle))
            self.assertEqual(len(diagnostic_rows), 1)
            self.assertEqual(diagnostic_rows[0]["op_name"], "svd")

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["result_status"], "partial")
            self.assertEqual(manifest["unique_success_rows"], 1)
            self.assertEqual(manifest["unique_non_success_rows"], 1)
            self.assertEqual(manifest["source_count"], 2)


if __name__ == "__main__":
    unittest.main()
