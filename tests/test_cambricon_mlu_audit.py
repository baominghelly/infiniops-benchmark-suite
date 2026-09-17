import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.cambricon_mlu_audit import (
    main,
    parse_memory_details,
    parse_snapshot,
    select_idle_device,
)


MEMORY_OUTPUT = """
Card 0
    Physical Memory Usage
        Total                           : 81920 MiB
        Used                            : 0 MiB
        Reserved                        : 1174 MiB
        Free                            : 80746 MiB
Card 1
    Physical Memory Usage
        Total                           : 81920 MiB
        Used                            : 16 MiB
        Reserved                        : 1174 MiB
        Free                            : 80730 MiB
"""


class CambriconMluAuditTests(unittest.TestCase):
    def test_parses_cnmon_devices_and_processes(self):
        output = """
| 0     /   MLU590-M9D   v1.0.3 |         0000:4C:00.0 | 3%                  0 |
| 1         45C     88 W/470 W |     0 MiB/81920 MiB | FULL          Default |
| 1     /   MLU590-M9D   v1.0.3 |         0000:51:00.0 | 0%                  0 |
| 0         43C     88 W/470 W |     16 MiB/81920 MiB | FULL          Default |

| Processes:                                                                   |
|  Card  MI  PID     Command Line                             MLU Memory Usage |
|==============================================================================|
|  0     /  12345 /torch/venv3/pytorch/bin/python                   16 MiB |
"""
        payload = parse_snapshot(output)

        self.assertEqual(payload["devices"]["0"]["utilization"], 3)
        self.assertEqual(payload["devices"]["0"]["memory_used_mib"], 0)
        self.assertEqual(payload["devices"]["0"]["memory_total_mib"], 81920)
        self.assertEqual(payload["devices"]["1"]["memory_used_mib"], 16)
        self.assertEqual(payload["devices"]["1"]["memory_total_mib"], 81920)
        self.assertEqual(payload["processes"][0]["device_id"], 0)
        self.assertEqual(payload["processes"][0]["pid"], 12345)
        self.assertEqual(payload["processes"][0]["memory_used_mib"], 16)

    def test_parses_detailed_physical_memory(self):
        details = parse_memory_details(MEMORY_OUTPUT)

        self.assertEqual(details["0"]["total"], 81920)
        self.assertEqual(details["0"]["used"], 0)
        self.assertEqual(details["0"]["reserved"], 1174)
        self.assertEqual(details["0"]["free"], 80746)
        self.assertEqual(details["1"]["free"], 80730)

    def test_selects_lowest_idle_mlu_across_samples(self):
        outputs = [
            """
| 0     /   MLU590-M9D   v1.0.3 | bus | 0%                  0 |
| 1         45C     88 W/470 W |     0 MiB/81920 MiB | FULL          Default |
| 1     /   MLU590-M9D   v1.0.3 | bus | 0%                  0 |
| 0         43C     88 W/470 W |     0 MiB/81920 MiB | FULL          Default |
| Processes: |
|  1     /  123 python 16 MiB |
""",
            """
| 0     /   MLU590-M9D   v1.0.3 | bus | 0%                  0 |
| 1         45C     88 W/470 W |     0 MiB/81920 MiB | FULL          Default |
| 1     /   MLU590-M9D   v1.0.3 | bus | 7%                  0 |
| 0         43C     88 W/470 W |     0 MiB/81920 MiB | FULL          Default |
| Processes: |
|  No running processes found |
""",
        ]
        with mock.patch(
            "scripts.cambricon_mlu_audit._cnmon_output",
            side_effect=outputs,
        ), mock.patch("scripts.cambricon_mlu_audit.time.sleep"), mock.patch(
            "scripts.cambricon_mlu_audit._cnmon_memory_output",
            return_value=MEMORY_OUTPUT,
        ):
            payload = select_idle_device(samples=2)

        self.assertEqual(payload["selected_mlu"], 0)
        self.assertEqual(payload["candidates"], ["0"])
        self.assertEqual(payload["busy"], ["1"])

    def test_free_memory_gate_and_occupied_mode_are_independent(self):
        output = """
| 0     /   MLU590-M9D   v1.0.3 | bus | 3%                  0 |
| 1         45C     88 W/470 W | 40960 MiB/81920 MiB | FULL          Default |
| Processes: |
|  0     /  123 python 16 MiB |
"""
        with mock.patch(
            "scripts.cambricon_mlu_audit._cnmon_output",
            return_value=output,
        ), mock.patch(
            "scripts.cambricon_mlu_audit._cnmon_memory_output",
            return_value="""
Card 0
    Physical Memory Usage
        Total                           : 81920 MiB
        Used                            : 40960 MiB
        Reserved                        : 0 MiB
        Free                            : 40960 MiB
""",
        ):
            exclusive = select_idle_device(
                samples=1,
                max_utilization=5,
                min_free_memory_percent=50,
            )
            occupied = select_idle_device(
                samples=1,
                max_utilization=5,
                min_free_memory_percent=50,
                allow_occupied=True,
            )

        self.assertIsNone(exclusive["selected_mlu"])
        self.assertEqual(occupied["selected_mlu"], 0)
        self.assertEqual(occupied["candidates"], ["0"])

    def test_watch_marks_foreign_allowed_without_flag(self):
        payload = {
            "timestamp": "now",
            "mlu": "0",
            "device": {},
            "processes": [{"pid": 123, "process_owner": "foreign_allowed"}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            done = Path(temporary) / "benchmark.done"
            foreign = Path(temporary) / "foreign-process.flag"
            done.write_text("exit_code=0\n", encoding="utf-8")
            with mock.patch(
                "scripts.cambricon_mlu_audit._json_payload",
                return_value=payload,
            ), contextlib.redirect_stdout(io.StringIO()):
                result = main(
                    [
                        "--mlu",
                        "0",
                        "--watch",
                        "--allow-occupied",
                        f"--done-file={done}",
                        f"--foreign-flag={foreign}",
                    ]
                )
            foreign_flag_exists = foreign.exists()

        self.assertEqual(result, 0)
        self.assertFalse(foreign_flag_exists)

    def test_shell_selection_preserves_physical_mlu_zero(self):
        payload = {
            "selected_mlu": 0,
            "candidates": ["0"],
            "busy": [],
            "rejected": [],
            "observations": [],
        }
        output = io.StringIO()
        with mock.patch(
            "scripts.cambricon_mlu_audit.select_idle_device",
            return_value=payload,
        ), contextlib.redirect_stdout(output):
            result = main(["--select", "--shell"])

        self.assertEqual(result, 0)
        self.assertIn("physical_mlu=0\n", output.getvalue())


if __name__ == "__main__":
    unittest.main()
