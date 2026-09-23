import unittest
from pathlib import Path
import subprocess
import tempfile
from unittest import mock


from ops import infinibenchctl as ctl


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


if __name__ == "__main__":
    unittest.main()
