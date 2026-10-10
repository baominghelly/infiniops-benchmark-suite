import unittest
from pathlib import Path
import subprocess
import tempfile
import datetime as dt
import contextlib
import io
import json
from unittest import mock


from ops import infinibenchctl as ctl

ROOT = Path(__file__).resolve().parents[1]
from ops import create_global_run as manifest
from ops import run_platform_worker as worker


class InfiniBenchControlContractTests(unittest.TestCase):
    def test_parser_exposes_five_operational_commands(self):
        parser = ctl.build_parser()
        args = parser.parse_args(["status"])
        self.assertEqual(args.command, "status")
        self.assertTrue(callable(args.func))

    def test_run_once_defaults_to_small_profile_and_all_dtypes(self):
        parser = ctl.build_parser()
        args = parser.parse_args(
            [
                "run-once",
                "--device",
                "nvidia-a100",
                "--device",
                "metax-c550",
            ]
        )
        self.assertEqual(args.profile, "decode-4k-small")
        self.assertEqual(args.dtypes, "float16,float32,bfloat16")

    def test_daily_control_covers_all_deployed_workers(self):
        self.assertEqual(
            ctl.DEVICES,
            (
                "nvidia-a100",
                "metax-c550",
                "ascend-910c",
                "moore-s5000",
                "iluvatar-tg150",
                "hygon-bw1100",
                "cambricon-mlu590",
            ),
        )
        self.assertEqual(
            sorted(ctl.REMOTE_WORKERS),
            sorted(ctl.DEVICES[1:]),
        )

    def test_daily_schedule_registers_every_worker_consistently(self):
        import re

        self.assertEqual(manifest.DEVICES, ctl.DEVICES)

        schedule = (ROOT / "ops" / "scheduled_daily.sh").read_text(encoding="utf-8")
        scheduled = set(re.findall(r"--device ([a-z0-9-]+)", schedule))
        self.assertEqual(scheduled, set(ctl.DEVICES))

        periodic = (ROOT / "ops" / "periodic_worker.sh").read_text(encoding="utf-8")
        for device in ctl.DEVICES:
            if device == "hygon-bw1100":
                self.assertIn("hygon-bw1100)\n    adapter=run-hygon-bw1100.sh", periodic)
            else:
                self.assertIn(f"{device}) adapter=", periodic)

        from ops import periodic_worker

        self.assertEqual(
            sorted(periodic_worker.ADAPTERS),
            sorted(worker.ADAPTERS),
        )
        hygon_remote = (
            ROOT / "ops" / "periodic_hygon_remote.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$host"', hygon_remote)
        self.assertIn("launch_node hygon251 hygon-bw1100-01", hygon_remote)
        self.assertIn("launch_node hygon252 hygon-bw1100-02", hygon_remote)
        self.assertIn("INFINIBENCH_DEVICE=hygon-bw1100", hygon_remote)
        self.assertIn("INFINIBENCH_NODE=", hygon_remote)
        self.assertIn("baoming_infiniops_hygon_dcu2", hygon_remote)
        self.assertIn("hygon-bw1100)\n    adapter=run-hygon-bw1100.sh", periodic)
        self.assertIn('benchmark_output = args.ops_root / args.device / execution_date', (ROOT / "ops" / "run_platform_worker.py").read_text(encoding="utf-8"))

        for device in ctl.DEVICES:
            remote = ctl.REMOTE_WORKERS.get(device)
            if not remote:
                continue
            adapter_name = Path(remote["adapter"]).name
            self.assertTrue(
                (ROOT / "hermes" / adapter_name).is_file(),
                adapter_name,
            )
            self.assertTrue(
                (ROOT / "ops" / "config" / f"{device}.yaml").is_file(),
                device,
            )

    def test_manifest_creator_captures_candidate_output_for_fallback(self):
        args = type("Args", (), {
            "global_run_id": "20260923-daily",
            "execution_time": "20260923-020000",
            "schedule_mode": "daily",
            "dtypes": "float16,float32,bfloat16",
            "profile": "decode-4k",
            "device": ["nvidia-a100"],
            "infiniops_ref": "master",
            "infinirt_ref": "master",
            "infiniops_commit": "a" * 40,
            "infinirt_commit": "b" * 40,
        })()
        completed = subprocess.CompletedProcess([], 0)
        with mock.patch.object(ctl, "run_command", return_value=completed) as runner:
            ctl.create_manifest(args)
        self.assertTrue(runner.call_args.kwargs["capture"])

    def test_pause_and_resume_use_shared_control_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(ctl, "CONTROL_ROOT", Path(directory)):
                args = type("Args", (), {})()
                ctl.command_pause(args)
                self.assertTrue(ctl.pause_marker().is_file())
                ctl.command_resume(args)
                self.assertFalse(ctl.pause_marker().is_file())

    def test_heartbeat_age_and_staleness_are_derived_from_manifest(self):
        recent = dt.datetime.now(dt.timezone.utc).isoformat()
        payload = {
            "state": "running",
            "heartbeat_at": recent,
            "heartbeat_stale_seconds": 120,
        }
        self.assertLessEqual(ctl.heartbeat_age_seconds(payload), 1)
        self.assertFalse(ctl.heartbeat_is_stale(payload))
        payload["heartbeat_at"] = "2026-09-23T00:00:00+00:00"
        self.assertGreater(ctl.heartbeat_age_seconds(payload), 120)
        self.assertTrue(ctl.heartbeat_is_stale(payload))

    def test_status_lists_each_active_worker_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for device in ctl.DEVICES:
                run = root / device / "20260923-daily"
                run.mkdir(parents=True)
                (run / "worker-manifest.json").write_text(
                    json.dumps({"state": "running", "heartbeat_at": dt.datetime.now(dt.timezone.utc).isoformat()}),
                    encoding="utf-8",
                )
            with mock.patch.object(
                ctl,
                "device_worker_root",
                side_effect=lambda device: root,
            ):
                with mock.patch.object(ctl, "pause_marker", return_value=root / "absent-pause"):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        ctl.command_status(type("Args", (), {})())
            payload = json.loads(output.getvalue())
            self.assertEqual(len(payload["active_workers"]), len(ctl.DEVICES))
            self.assertEqual(
                [entry["device"] for entry in payload["active_workers"]],
                sorted(ctl.DEVICES),
            )
            self.assertEqual(len(payload["workers"]["nvidia-a100"]), 1)
            self.assertEqual(len(payload["workers"]["metax-c550"]), 1)
            self.assertEqual(len(payload["workers"]["ascend-910c"]), 1)
            self.assertEqual(len(payload["workers"]["moore-s5000"]), 1)
            self.assertEqual(len(payload["workers"]["iluvatar-tg150"]), 1)


if __name__ == "__main__":
    unittest.main()
