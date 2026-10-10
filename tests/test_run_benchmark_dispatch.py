from pathlib import Path
import unittest


class RunBenchmarkDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.script = (root / "scripts" / "run_benchmark.sh").read_text(
            encoding="utf-8"
        )

    def test_dispatches_supported_platforms_and_bw1100_node_aliases(self):
        expected = {
            "nvidia-a100": "nvidia-a100",
            "metax-c550": "metax-c550",
            "ascend-910c": "ascend-910c",
            "moore-s5000": "moore-s5000",
            "iluvatar-tg150": "iluvatar-tg150",
            "cambricon-mlu590": "cambricon-mlu590",
            "hygon-bw1000": "hygon-bw1000",
            "hygon-bw1100": "hygon-bw1100",
            "bw1100-01": "hygon-bw1100",
            "bw1100-02": "hygon-bw1100",
        }
        for device, output_device in expected.items():
            with self.subTest(device=device):
                self.assertIn(device, self.script)
                self.assertIn(f"device_id={output_device}", self.script)
        self.assertIn("runner=run_hygon_benchmark.sh", self.script)
        self.assertNotIn("device_id=hygon-bw1100-01", self.script)
        self.assertNotIn("device_id=hygon-bw1100-02", self.script)

    def test_uses_canonical_operator_output_root_and_forwards_gpu_args(self):
        self.assertIn(
            'output_root="${OUTPUT_ROOT:-/data-aisoft/infinibench_data/ops}"',
            self.script,
        )
        self.assertIn('export DEVICE_ID="$device_id"', self.script)
        self.assertIn('export OUTPUT_ROOT="$output_root"', self.script)
        self.assertIn('forward_args+=("$@")', self.script)
        self.assertIn('exec "$runner_path" "${forward_args[@]}"', self.script)

    def test_bw1100_nodes_share_a_platform_output_directory(self):
        self.assertIn("bw1100|hygon-bw1100)", self.script)
        self.assertIn(
            "bw1100-01|hygon-bw1100-01|bw1100-02|hygon-bw1100-02)",
            self.script,
        )
        self.assertIn("baoming_infiniops_hygon_dcu2", self.script)

    def test_supports_discovery_and_dry_print_without_launching_runner(self):
        self.assertIn("--list", self.script)
        self.assertIn("--print", self.script)
        self.assertIn('printf \'DEVICE_ID=%s\\n\' "$device_id"', self.script)
        self.assertIn('printf \'RUNNER=%s\\n\' "$runner_path"', self.script)
        self.assertIn('printf \'OUTPUT_DIR=%s/%s/<run-date>\\n\'', self.script)


if __name__ == "__main__":
    unittest.main()
