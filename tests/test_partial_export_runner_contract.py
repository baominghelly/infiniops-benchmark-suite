from pathlib import Path
import unittest


class PartialExportRunnerContractTests(unittest.TestCase):
    def test_all_platform_runners_export_partial_results_on_failure(self):
        root = Path(__file__).resolve().parents[1]
        runners = [
            "run_a100_benchmark.sh",
            "run_ascend_benchmark.sh",
            "run_cambricon_benchmark.sh",
            "run_hygon_benchmark.sh",
            "run_iluvatar_benchmark.sh",
            "run_metax_benchmark.sh",
            "run_moore_benchmark.sh",
        ]
        for runner in runners:
            with self.subTest(runner=runner):
                script = (root / "scripts" / runner).read_text(encoding="utf-8")
                self.assertIn('partial_exporter="$container_tools_root/scripts/export_partial_results.py"', script)
                self.assertIn('docker exec "$container" test -f "$partial_exporter"', script)
                self.assertIn('if [[ "$exit_code" -ne 0 || -s "$foreign_flag" ]]', script)
                self.assertIn("results.jsonl", script)
                self.assertIn(".partial.csv", script)


if __name__ == "__main__":
    unittest.main()
