from pathlib import Path
import unittest


class RunMetaXBenchmarkContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_metax_benchmark.sh"
        ).read_text(encoding="utf-8")
        cls.guide = (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "run_metax_benchmark.md"
        ).read_text(encoding="utf-8")

    def test_backfill_directory_is_deterministic_by_date_pair(self):
        self.assertIn('execution_date="${execution_time:0:8}"', self.script)
        self.assertIn(
            'run_date="backfill-${backfill_date//-/}-${execution_date}"',
            self.script,
        )
        self.assertIn(
            "backfill-<YYYYMMDD>-<EXECUTION_DATE>/",
            self.guide,
        )

    def test_existing_backfill_directory_is_not_reused_silently(self):
        self.assertIn('reuse_run_dir="${REUSE_RUN_DIR:-0}"', self.script)
        self.assertIn("output directory already exists and is non-empty", self.script)
        self.assertIn("REUSE_RUN_DIR=1 to replace it", self.script)
        self.assertIn('case "$output_container" in', self.script)
        self.assertIn("refusing to reset untrusted output path", self.script)
        self.assertIn("To replace a failed", self.guide)
        self.assertIn("set `REUSE_RUN_DIR=1`", self.guide)


if __name__ == "__main__":
    unittest.main()
