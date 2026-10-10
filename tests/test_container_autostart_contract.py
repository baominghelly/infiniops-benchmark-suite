from pathlib import Path
import unittest


class ContainerAutostartContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.helper = (root / "scripts" / "ensure_benchmark_container.sh").read_text(
            encoding="utf-8"
        )
        cls.runners = {
            path.name: path.read_text(encoding="utf-8")
            for path in (root / "scripts").glob("run_*_benchmark.sh")
            if path.name
            not in {
                "run_benchmark.sh",
                # Legacy host-specific recovery runner; the supported TG150
                # public adapter is run_iluvatar_benchmark.sh.
                "run_tianshu58_benchmark.sh",
            }
        }

    def test_all_platform_runners_use_container_preflight(self):
        expected = {
            "run_a100_benchmark.sh",
            "run_metax_benchmark.sh",
            "run_ascend_benchmark.sh",
            "run_moore_benchmark.sh",
            "run_iluvatar_benchmark.sh",
            "run_cambricon_benchmark.sh",
            "run_hygon_benchmark.sh",
        }
        self.assertEqual(set(self.runners), expected)
        for runner, script in self.runners.items():
            with self.subTest(runner=runner):
                self.assertIn(
                    '"$host_tools_root/scripts/ensure_benchmark_container.sh" "$container"',
                    script,
                )
                self.assertNotIn('if ! docker inspect "$container" >/dev/null', script)

    def test_helper_starts_only_existing_created_or_exited_container(self):
        self.assertIn('auto_start="${AUTO_START_CONTAINER:-1}"', self.helper)
        self.assertIn('case "$auto_start" in', self.helper)
        self.assertIn('created|exited', self.helper)
        self.assertIn('docker start "$container"', self.helper)
        self.assertIn(
            '[[ "$(docker inspect -f \'{{.State.Running}}\' "$container")" == "true" ]]',
            self.helper,
        )
        self.assertIn("container not found", self.helper)
        self.assertIn("non-runnable state", self.helper)
        self.assertIn("did not reach running state", self.helper)

    def test_autostart_can_be_disabled_for_diagnostics(self):
        self.assertIn('AUTO_START_CONTAINER=0', self.helper)
        self.assertIn("to disable startup", self.helper)


if __name__ == "__main__":
    unittest.main()
