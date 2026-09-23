import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from ops import run_platform_worker as worker


ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"


class InfiniBenchOpsSystemTests(unittest.TestCase):
    def create_run(self, root: Path) -> tuple[Path, str]:
        run_id = "20260922-manual"
        runs_root = root / "ops" / ".runs"
        subprocess.run(
            [
                sys.executable,
                str(OPS / "create_global_run.py"),
                "--runs-root",
                str(runs_root),
                "--global-run-id",
                run_id,
                "--execution-time",
                "20260922-200000",
                "--infiniops-commit",
                "a" * 40,
                "--infinirt-commit",
                "b" * 40,
            ],
            check=True,
        )
        return runs_root / run_id / "global-manifest.json", run_id

    def test_create_global_run_is_atomic_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, run_id = self.create_run(root)
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["global_run_id"], run_id)
            self.assertEqual(payload["state"], "ready")
            self.assertEqual(payload["source"]["infiniops_commit"], "a" * 40)
            self.assertEqual(payload["source"]["infinirt_commit"], "b" * 40)
            self.create_run(root)

    def test_worker_dry_run_reads_manifest_and_does_not_create_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, run_id = self.create_run(root)
            adapter = root / "adapter.sh"
            adapter.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(OPS / "run_platform_worker.py"),
                    "--global-run-id",
                    run_id,
                    "--device",
                    "metax-c550",
                    "--runs-root",
                    str(root / "ops/.runs"),
                    "--ops-root",
                    str(root / "ops"),
                    "--worker-root",
                    str(root / "workers"),
                    "--adapter",
                    str(adapter),
                    "--dry-run",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertIn("dry_run=true", result.stdout)
            self.assertIn("infiniops_commit=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", result.stdout)
            self.assertFalse((root / "workers").exists())

    def test_platform_wrappers_forward_frozen_refs(self):
        nvidia = (ROOT / "hermes" / "run-nvidia-a100-benchmark.sh").read_text(
            encoding="utf-8"
        )
        metax = (ROOT / "hermes" / "run-metax-c550-weekly.sh").read_text(
            encoding="utf-8"
        )
        for script in (nvidia, metax):
            self.assertIn('infiniops_ref="${INFINIOPS_REF:-}"', script)
            self.assertIn('infinirt_ref="${INFINIRT_REF:-}"', script)
            self.assertIn('INFINIOPS_REF="$infiniops_ref"', script)
            self.assertIn('INFINIRT_REF="$infinirt_ref"', script)

    def test_failure_policy_allows_retry_only_for_known_transient_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            console = root / "benchmark-console.log"
            output = root / "benchmark-output"
            output.mkdir()
            console.write_text(
                "=== container preflight ===\n"
                "AssertionError\nassert torch.cuda.is_available()\n",
                encoding="utf-8",
            )
            policy = worker.classify_failure(console, output, 1)
            self.assertEqual(policy["failure_code"], "container_cuda_lost")
            self.assertTrue(policy["auto_retry"])
            self.assertTrue(policy["allow_replace_existing_output"])
            self.assertEqual(policy["recommended_action"], "restart_benchmark_container_then_retry_without_rebuild")

    def test_failure_policy_preserves_checkpoints_and_blocks_auto_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            console = root / "benchmark-console.log"
            output = root / "benchmark-output" / "float16" / "decode-4k"
            output.mkdir(parents=True)
            console.write_text(
                "AssertionError\nassert torch.cuda.is_available()\n",
                encoding="utf-8",
            )
            (output / "results.jsonl").write_text("{}\n", encoding="utf-8")
            policy = worker.classify_failure(console, root / "benchmark-output", 1)
            self.assertFalse(policy["auto_retry"])
            self.assertTrue(policy["preserve_partial_results"])
            self.assertFalse(policy["allow_replace_existing_output"])
            self.assertEqual(policy["checkpoint_result_count"], 1)

    def test_unknown_failure_requires_manual_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            console = root / "benchmark-console.log"
            output = root / "benchmark-output"
            output.mkdir()
            console.write_text("unexpected algebra error\n", encoding="utf-8")
            policy = worker.classify_failure(console, output, 1)
            self.assertEqual(policy["failure_code"], "benchmark_failure")
            self.assertFalse(policy["auto_retry"])
            self.assertEqual(policy["recommended_action"], "inspect_console_and_diagnostics_manually")

    def test_worker_manifest_contract_declares_heartbeat_and_failure_policy(self):
        script = (OPS / "run_platform_worker.py").read_text(encoding="utf-8")
        self.assertIn("HEARTBEAT_INTERVAL_SECONDS = 30", script)
        self.assertIn("HEARTBEAT_STALE_SECONDS = 120", script)
        self.assertIn('snapshot["heartbeat_at"] = utc_now()', script)
        self.assertIn('"failure_policy": failure_policy', script)
        self.assertIn('"auto_retry": failure_policy["auto_retry"]', script)


if __name__ == "__main__":
    unittest.main()
