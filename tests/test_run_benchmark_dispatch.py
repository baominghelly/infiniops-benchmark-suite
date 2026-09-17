from pathlib import Path
import unittest


class RunBenchmarkDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.script = (root / "scripts" / "run_benchmark.sh").read_text(
            encoding="utf-8"
        )

    def test_dispatches_all_seven_supported_devices(self):
        expected = {
            "nvidia-a100": "run_a100_benchmark.sh",
            "metax-c550": "run_metax_benchmark.sh",
            "ascend-910c": "run_ascend_benchmark.sh",
            "moore-s5000": "run_moore_benchmark.sh",
            "iluvatar-tg150": "run_iluvatar_benchmark.sh",
            "cambricon-mlu590": "run_cambricon_benchmark.sh",
            "hygon-bw1000": "run_hygon_benchmark.sh",
        }
        for device, runner in expected.items():
            with self.subTest(device=device):
                self.assertIn(f"{device})\n    device_id={device}", self.script)
                self.assertIn(f"runner={runner}", self.script)

    def test_uses_canonical_operator_output_root_and_forwards_gpu_args(self):
        self.assertIn(
            'output_root="${OUTPUT_ROOT:-/data-aisoft/infinibench_data/ops}"',
            self.script,
        )
        self.assertIn('export DEVICE_ID="$device_id"', self.script)
        self.assertIn('export OUTPUT_ROOT="$output_root"', self.script)
        self.assertIn('forward_args+=("$@")', self.script)
        self.assertIn('exec "$runner_path" "${forward_args[@]}"', self.script)

    def test_supports_discovery_and_dry_print_without_launching_runner(self):
        self.assertIn("--list", self.script)
        self.assertIn("--print", self.script)
        self.assertIn("printf 'DEVICE_ID=%s\\n' \"$device_id\"", self.script)
        self.assertIn("printf 'RUNNER=%s\\n' \"$runner_path\"", self.script)
        self.assertIn("printf 'OUTPUT_DIR=%s/%s/<run-date>\\n'", self.script)


if __name__ == "__main__":
    unittest.main()
