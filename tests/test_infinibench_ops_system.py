import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
