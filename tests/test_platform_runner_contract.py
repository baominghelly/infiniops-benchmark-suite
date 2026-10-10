import unittest
from pathlib import Path

from scripts.full_operator_benchmark import build_parser
from ops import run_platform_worker


ROOT = Path(__file__).resolve().parents[1]


class PlatformRunnerContractTests(unittest.TestCase):
    def test_all_pilot_runners_pass_platform_to_harness(self):
        runners = {
            "nvidia-a100": "run_a100_benchmark.sh",
            "metax-c550": "run_metax_benchmark.sh",
            "ascend-910c": "run_ascend_benchmark.sh",
            "moore-s5000": "run_moore_benchmark.sh",
            "iluvatar-tg150": "run_iluvatar_benchmark.sh",
            "hygon-bw1100": "run_hygon_benchmark.sh",
        }
        for platform, runner in runners.items():
            with self.subTest(platform=platform):
                script = (ROOT / "scripts" / runner).read_text(encoding="utf-8")
                self.assertIn('--platform "$device_id"', script)

    def test_all_platform_runners_support_quick_min_run_time(self):
        runners = [
            "run_a100_benchmark.sh",
            "run_metax_benchmark.sh",
            "run_ascend_benchmark.sh",
            "run_moore_benchmark.sh",
            "run_iluvatar_benchmark.sh",
            "run_cambricon_benchmark.sh",
            "run_hygon_benchmark.sh",
        ]
        for runner in runners:
            with self.subTest(runner=runner):
                script = (ROOT / "scripts" / runner).read_text(encoding="utf-8")
                self.assertIn('min_run_time="${MIN_RUN_TIME:-0.02}"', script)
                self.assertIn('invalid MIN_RUN_TIME:', script)
                self.assertIn('--min-run-time "$min_run_time"', script)
                self.assertIn('benchmark_iterations="${BENCHMARK_ITERATIONS:-0}"', script)
                self.assertIn('invalid BENCHMARK_ITERATIONS:', script)
                self.assertIn('--benchmark-iterations "$benchmark_iterations"', script)

    def test_metax_runner_supports_worker_batching(self):
        script = (ROOT / "scripts" / "run_metax_benchmark.sh").read_text(encoding="utf-8")
        self.assertIn('worker_batch_size="${WORKER_BATCH_SIZE:-1}"', script)
        self.assertIn(
            'worker_batch_timeout="${WORKER_BATCH_TIMEOUT:-$((20 * worker_batch_size))}"',
            script,
        )
        self.assertIn("invalid WORKER_BATCH_SIZE:", script)
        self.assertIn("invalid WORKER_BATCH_TIMEOUT:", script)
        self.assertIn('--worker-batch-size "$worker_batch_size"', script)
        self.assertIn('--batch-timeout "$worker_batch_timeout"', script)

    def test_hygon_daily_worker_uses_logical_platform_and_physical_nodes(self):
        env = run_platform_worker.PLATFORM_ENV["hygon-bw1100"]
        self.assertEqual(env["DEVICE"], "hygon-bw1100")
        self.assertEqual(env["HYGON_CUDA_ROOT"], "/opt/dtk/cuda/cuda-12")
        self.assertEqual(
            run_platform_worker.HYGON_NODES_BY_HOST["dcu1"],
            "hygon-bw1100-01",
        )
        self.assertEqual(
            run_platform_worker.HYGON_NODES_BY_HOST["dcu2"],
            "hygon-bw1100-02",
        )
        self.assertEqual(
            run_platform_worker.HYGON_CONTAINERS_BY_NODE["hygon-bw1100-02"],
            "baoming_infiniops_hygon_dcu2",
        )

    def test_harness_accepts_explicit_platform(self):
        args = build_parser().parse_args(
            ["--metadata", "metadata.json", "--platform", "moore-s5000"]
        )
        self.assertEqual(args.platform, "moore-s5000")


if __name__ == "__main__":
    unittest.main()
