from pathlib import Path
import unittest


class CommitMetadataContractTests(unittest.TestCase):
    def test_all_platform_exporters_receive_both_source_commits(self):
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
                self.assertIn('--infiniops-commit "$sha"', script)
                self.assertIn('--infinirt-commit "$infinirt_sha"', script)


if __name__ == "__main__":
    unittest.main()
