import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from hermes.nvidia_a100_doctor import (
    apply_playbook,
    classify_failure,
    find_latest_run,
    prune_old_hermes_logs,
)


class NvidiaA100DoctorTests(unittest.TestCase):
    def make_run(self, text=None, **outcome):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "20260921"
            run.mkdir()
            (run / "failure.json").write_text(
                json.dumps(outcome), encoding="utf-8"
            )
            if text is not None:
                (run / "attempt.log").write_text(text, encoding="utf-8")
            yield root, run

    def test_classifies_container_cuda_lost_from_probe_context(self):
        for _, run in self.make_run(
            final_exit_code=1,
            final_console="attempt.log",
            benchmark_output="/data/aisoft/out/20260921",
        ):
            result = classify_failure(
                run,
                {
                    "host_gpu_ok": True,
                    "container_cuda_ok": False,
                    "container_running": True,
                    "benchmark_process_running": False,
                },
            )
        self.assertEqual(result["failure_code"], "container_cuda_lost")
        self.assertFalse(result["needs_llm"])

    def test_source_dirty_has_priority_and_needs_human_review(self):
        for _, run in self.make_run(
            final_exit_code=2,
            final_console="attempt.log",
            benchmark_output="/out/20260921",
            text="repository has local changes; refusing to overwrite",
        ):
            result = classify_failure(run, {})
        self.assertEqual(result["failure_code"], "source_dirty")
        self.assertEqual(result["action"], "capture_source_dirty_state")
        self.assertFalse(result["needs_llm"])

    def test_classifies_gpu_gate_and_foreign_process_failures(self):
        cases = {
            "no idle NVIDIA GPU found": "no_idle_gpu",
            "physical NVIDIA GPU 0 has a running process": "foreign_process",
            "process_owner=foreign pid=123": "foreign_process",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                for _, run in self.make_run(
                    final_exit_code=2,
                    final_console="attempt.log",
                    benchmark_output="/out/20260921",
                    text=text,
                ):
                    self.assertEqual(
                        classify_failure(run, {})["failure_code"], expected
                    )

    def test_classifies_benchmark_container_down_and_docker_down(self):
        cases = [
            ({"docker_available": True, "container_running": False}, "benchmark_container_down"),
            ({"docker_available": False, "container_running": False}, "docker_down"),
        ]
        for context, expected in cases:
            with self.subTest(expected=expected):
                for _, run in self.make_run(
                    final_exit_code=1,
                    final_console="attempt.log",
                    benchmark_output="/out/20260921",
                ):
                    self.assertEqual(
                        classify_failure(run, context)["failure_code"], expected
                    )

    def test_unknown_failure_requests_llm(self):
        for _, run in self.make_run(
            final_exit_code=7,
            final_console="attempt.log",
            benchmark_output="/out/20260921",
            text="completely unexpected compiler message",
        ):
            result = classify_failure(run, {})
        self.assertEqual(result["failure_code"], "unknown")
        self.assertTrue(result["needs_llm"])

    def test_success_is_not_treated_as_blocked_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "20260921"
            run.mkdir()
            result = classify_failure(run, {})
            self.assertEqual(result["failure_code"], "none")
            applied = apply_playbook(
                run,
                result,
                {},
                apply=True,
                container="unused",
                runs_root=run.parent,
            )
            self.assertEqual(applied["status"], "no_failure")
            self.assertFalse(applied["needs_llm"])

    def test_disk_full_dry_run_does_not_delete_files(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "20260921"
            old = Path(directory) / "20260801"
            old.mkdir()
            run.mkdir()
            log = old / "attempt.log"
            log.write_text("x" * 100, encoding="utf-8")
            old_time = time.time() - 31 * 86400
            os.utime(log, (old_time, old_time))
            result = apply_playbook(
                run,
                {
                    "failure_code": "disk_full",
                    "action": "prune_hermes_run_logs",
                },
                {"free_bytes": 1},
                apply=False,
                container="unused",
                runs_root=Path(directory),
            )
            self.assertEqual(result["status"], "dry_run")
            self.assertTrue(log.exists())

    def test_find_latest_run_ignores_non_run_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "20260920").mkdir()
            (root / "20260921").mkdir()
            (root / "nvidia-a100.lock").write_text("", encoding="utf-8")
            self.assertEqual(find_latest_run(root), root / "20260921")

    def test_prune_only_old_files_under_runs_root_and_respects_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            old = root / "20260801"
            current = root / "20260921"
            old.mkdir(parents=True)
            current.mkdir(parents=True)
            (old / "attempt.log").write_text("x" * 100, encoding="utf-8")
            (current / "attempt.log").write_text("x" * 100, encoding="utf-8")
            removed = prune_old_hermes_logs(
                runs_root=root,
                current_run=current,
                retain_days=30,
                max_bytes=50,
                now_epoch=2_000_000_000,
            )
            self.assertEqual(removed["files_removed"], 0)
            self.assertTrue((old / "attempt.log").exists())

            removed = prune_old_hermess_logs_cap_100(
                old, current, max_bytes=100
            )
            self.assertEqual(removed["files_removed"], 1)
            self.assertFalse((old / "attempt.log").exists())
            self.assertTrue((current / "attempt.log").exists())


def prune_old_hermess_logs_cap_100(old, current, max_bytes):
    return prune_old_hermes_logs(
        runs_root=old.parent,
        current_run=current,
        retain_days=0,
        max_bytes=max_bytes,
    )


if __name__ == "__main__":
    unittest.main()
