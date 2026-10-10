import unittest
from pathlib import Path


class HermesWorkflowContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.workflow = (root / "hermes" / "run-nvidia-a100-benchmark.sh").read_text(
            encoding="utf-8"
        )
        cls.doctor_entry = (
            root / "hermes" / "run-nvidia-a100-doctor.sh"
        ).read_text(encoding="utf-8")
        cls.cambricon_collector = (
            root / "hermes" / "run-cambricon-result-collector.sh"
        ).read_text(encoding="utf-8")

    def test_workflow_records_final_machine_readable_outcome(self):
        self.assertIn("write_outcome()", self.workflow)
        self.assertIn('outcome_file="$run_root/failure.json"', self.workflow)
        self.assertIn('outcome_file="$run_root/success.json"', self.workflow)
        self.assertIn('"final_console": sys.argv[3]', self.workflow)
        self.assertIn('"execution_date": sys.argv[5]', self.workflow)

    def test_outcome_is_written_for_success_and_both_failure_paths(self):
        self.assertIn("write_outcome success 0", self.workflow)
        self.assertIn("write_outcome failure", self.workflow)
        self.assertIn('console="$run_root/attempt-1-exclusive.log"', self.workflow)
        self.assertIn('console="$run_root/attempt-2-shared.log"', self.workflow)

    def test_new_outcome_replaces_stale_counterpart(self):
        self.assertIn('rm -f -- "$run_root/failure.json"', self.workflow)
        self.assertIn('rm -f -- "$run_root/success.json"', self.workflow)

    def test_cambricon_collector_only_archives_complete_result_runs(self):
        self.assertIn("benchmark.done", self.cambricon_collector)
        self.assertIn("infiniops-*.csv", self.cambricon_collector)
        self.assertIn("completed_without_result_csv", self.cambricon_collector)
        self.assertIn(".collector-success.json", self.cambricon_collector)

    def test_cambricon_collector_uses_staging_and_never_overwrites(self):
        self.assertIn("mktemp -d \"$dest_root/.collector-staging-", self.cambricon_collector)
        self.assertIn('destination_already_exists', self.cambricon_collector)
        self.assertIn('mv "$stage" "$final"', self.cambricon_collector)
        self.assertNotIn("--delete", self.cambricon_collector)

    def test_cambricon_collector_uses_dedicated_persistent_key(self):
        self.assertIn("/opt/data/ssh/cambricon_result_collector", self.cambricon_collector)
        self.assertIn("StrictHostKeyChecking=yes", self.cambricon_collector)

    def test_host_invocation_reuses_running_hermes_account(self):
        self.assertIn("docker exec -i -u hermes", self.cambricon_collector)
        self.assertIn("baoming_hermes_agent", self.cambricon_collector)
        self.assertNotIn("docker run --rm", self.cambricon_collector)

    def test_doctor_entry_applies_deterministic_recovery_only(self):
        self.assertIn("nvidia_a100_doctor.py", self.doctor_entry)
        self.assertIn("--apply", self.doctor_entry)
        self.assertIn("--execute-retry", self.doctor_entry)
        self.assertNotIn("hermes -z", self.doctor_entry)

    def test_new_outcome_replaces_stale_counterpart(self):
        self.assertIn('rm -f -- "$run_root/failure.json"', self.workflow)
        self.assertIn('rm -f -- "$run_root/success.json"', self.workflow)


if __name__ == "__main__":
    unittest.main()
