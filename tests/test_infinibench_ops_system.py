import json
import subprocess
import sys
import tempfile
import unittest
import datetime as dt
from pathlib import Path
from ops import run_platform_worker as worker
from ops import periodic_worker


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
        for name in (
            "run-ascend-910c.sh",
            "run-iluvatar-tg150.sh",
            "run-moore-s5000.sh",
            "run-hygon-bw1100-01.sh",
            "run-hygon-bw1100.sh",
        ):
            script = (ROOT / "hermes" / name).read_text(encoding="utf-8")
            self.assertIn('export DEVICE="${BENCHMARK_DEVICE:-', script)
            self.assertIn('runner="${BENCHMARK_RUNNER:-', script)
        hygon_legacy = (ROOT / "hermes" / "run-hygon-bw1100-01.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('export DEVICE="${BENCHMARK_DEVICE:-hygon-bw1100}"', hygon_legacy)
        hygon = (ROOT / "hermes" / "run-hygon-bw1100.sh").read_text(encoding="utf-8")
        self.assertIn('export DEVICE="${BENCHMARK_DEVICE:-hygon-bw1100}"', hygon)

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
            self.assertEqual(policy["max_attempts"], 16)
            self.assertEqual(policy["retry_delay_base_seconds"], 300)
            self.assertEqual(policy["retry_delay_multiplier"], 2.0)
            self.assertEqual(policy["retry_delay_cap_seconds"], 3600)
            self.assertEqual(policy["retry_window_seconds"], 20 * 60 * 60)

    def test_failure_policy_preserves_checkpoints_and_uses_resume_retry(self):
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
            self.assertTrue(policy["auto_retry"])
            self.assertTrue(policy["preserve_partial_results"])
            self.assertFalse(policy["allow_replace_existing_output"])
            self.assertTrue(policy["resume_existing_output"])
            self.assertEqual(policy["checkpoint_result_count"], 1)

    def test_failure_policy_reads_platform_foreign_process_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            console = root / "benchmark-console.log"
            output = root / "benchmark-output" / "bfloat16" / "decode-4k"
            output.mkdir(parents=True)
            console.write_text("missing_result=.../results.json\n", encoding="utf-8")
            (root / "benchmark-output" / "benchmark.done").write_text(
                "exit_code=1\nforeign_process_seen=1\n",
                encoding="utf-8",
            )
            (root / "benchmark-output" / "foreign-process.flag").write_text(
                "3850197\n",
                encoding="utf-8",
            )
            (output / "results.jsonl").write_text("{}\n", encoding="utf-8")
            policy = worker.classify_failure(console, root / "benchmark-output", 3)
            self.assertEqual(policy["failure_code"], "gpu_busy")
            self.assertTrue(policy["auto_retry"])
            self.assertTrue(policy["preserve_partial_results"])
            self.assertTrue(policy["resume_existing_output"])
            self.assertIn("foreign_process_seen=1", policy["console_tail"])

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

    def test_periodic_retry_uses_capped_exponential_backoff(self):
        delays = [
            periodic_worker.retry_delay_seconds(
                {
                    "attempts": attempt,
                    "failure_policy": {
                        "retry_delay_base_seconds": 300,
                        "retry_delay_multiplier": 2.0,
                        "retry_delay_cap_seconds": 3600,
                    },
                }
            )
            for attempt in range(1, 7)
        ]
        self.assertEqual(delays, [300, 600, 1200, 2400, 3600, 3600])

    def test_periodic_retry_window_blocks_late_new_attempts(self):
        created = "2026-10-08T00:00:00+00:00"
        now = periodic_worker.parse_time("2026-10-08T20:00:00+00:00")
        manifest = {"schedule": {"created_at": created}}
        worker_state = {"failure_policy": {"retry_window_seconds": 72000}}
        self.assertFalse(
            periodic_worker.retry_window_allows_new_attempt(
                manifest, worker_state, now
            )
        )
        worker_state["failure_policy"]["retry_window_seconds"] = 72001
        self.assertTrue(
            periodic_worker.retry_window_allows_new_attempt(
                manifest, worker_state, now
            )
        )

    def test_hygon_logical_platform_uses_shared_state_and_failover(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / ".runs" / "20261009-daily"
            workers = root / ".workers" / "hygon-bw1100" / "20261009-daily"
            runs.mkdir(parents=True)
            workers.mkdir(parents=True)
            now = dt.datetime.now(dt.timezone.utc)
            (runs / "global-manifest.json").write_text(
                json.dumps(
                    {
                        "global_run_id": "20261009-daily",
                        "state": "ready",
                        "schedule": {"created_at": now.isoformat()},
                        "platforms": ["hygon-bw1100"],
                    }
                ),
                encoding="utf-8",
            )
            fresh = {
                "state": "running",
                "heartbeat_at": now.isoformat(),
                "heartbeat_stale_seconds": 120,
                "attempts": 1,
            }
            (workers / "worker-manifest.json").write_text(
                json.dumps(fresh), encoding="utf-8"
            )
            self.assertIsNone(
                periodic_worker.select_candidate(
                    runs.parent, root / ".control", workers.parent.parent, "hygon-bw1100", 48
                )
            )
            fresh["heartbeat_at"] = "2026-10-08T00:00:00+00:00"
            (workers / "worker-manifest.json").write_text(
                json.dumps(fresh), encoding="utf-8"
            )
            self.assertEqual(
                periodic_worker.select_candidate(
                    runs.parent, root / ".control", workers.parent.parent, "hygon-bw1100", 48
                ),
                ("20261009-daily", "worker heartbeat is stale"),
            )

    def test_worker_manifest_contract_declares_heartbeat_and_failure_policy(self):
        script = (OPS / "run_platform_worker.py").read_text(encoding="utf-8")
        self.assertIn("HEARTBEAT_INTERVAL_SECONDS = 30", script)
        self.assertIn("HEARTBEAT_STALE_SECONDS = 120", script)
        self.assertIn('snapshot["heartbeat_at"] = utc_now()', script)
        self.assertIn('"failure_policy": failure_policy', script)
        self.assertIn('"auto_retry": failure_policy["auto_retry"]', script)


if __name__ == "__main__":
    unittest.main()
