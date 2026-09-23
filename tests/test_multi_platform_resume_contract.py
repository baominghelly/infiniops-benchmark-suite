import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MultiPlatformResumeContractTests(unittest.TestCase):
    def test_platform_runners_accept_resume_without_replacing_output(self):
        for name in (
            "run_ascend_benchmark.sh",
            "run_moore_benchmark.sh",
            "run_iluvatar_benchmark.sh",
        ):
            script = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            with self.subTest(runner=name):
                self.assertIn('resume_run="${RESUME:-0}"', script)
                self.assertIn('REUSE_RUN_DIR=1 and RESUME=1 cannot be combined', script)
                self.assertIn('resume_run" !=', script)
                self.assertIn("--resume", script)


if __name__ == "__main__":
    unittest.main()
