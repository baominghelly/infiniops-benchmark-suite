import unittest
from unittest import mock
import contextlib
import io

from scripts.hygon_gpu_audit import main, processes


class HygonGpuAuditTests(unittest.TestCase):
    def test_parses_selected_hcu_process_records(self):
        output = """PIDs for KFD processes:

PID: 123
\tPASID: 1
\tHCU Node(Include CPU sort): ['9']
\tHCU Index: ['1']
\tVRAM USED(MiB): 53695

PID: 456
\tPASID: 2
\tHCU Node(Include CPU sort):
\tHCU Index:
\tVRAM USED(MiB): 2
"""
        with mock.patch("scripts.hygon_gpu_audit._rocm_smi_output", return_value=output):
            records = processes()

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["pid"], 123)
        self.assertEqual(records[0]["hcu_indices"], ["1"])
        self.assertEqual(records[0]["vram_used_mib"], 53695)
        self.assertEqual(records[1]["hcu_indices"], [])

    def test_select_reports_first_idle_hcu(self):
        def output(arguments=None):
            if arguments == ["--showproductname"]:
                return "GPU[0]: BW\nGPU[1]: BW\nGPU[2]: BW\n"
            if arguments == ["--showuse"]:
                return "GPU[0]: 0%\nGPU[1]: 7%\nGPU[2]: 2%\n"
            if arguments == ["--showmeminfo", "vram"]:
                return (
                    "GPU[0]: vram total: 65536 MiB, vram used: 32768 MiB\n"
                    "GPU[1]: vram total: 65536 MiB, vram used: 65536 MiB\n"
                    "GPU[2]: vram total: 65536 MiB, vram used: 6553 MiB\n"
                )
            return """PID: 123
	HCU Index: ['0']
"""

        stdout = io.StringIO()
        with mock.patch("scripts.hygon_gpu_audit._rocm_smi_output", side_effect=output):
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["--select"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            stdout.getvalue().splitlines(),
            ["2", "2", "0", "0(busy);1(utilization=7)"],
        )

    def test_select_can_allow_an_occupied_gpu_for_smoke_runs(self):
        def output(arguments=None):
            if arguments == ["--showproductname"]:
                return "GPU[0]: BW\nGPU[1]: BW\n"
            if arguments == ["--showuse"]:
                return "GPU[0]: 0%\nGPU[1]: 0%\n"
            if arguments == ["--showmeminfo", "vram"]:
                return "GPU[0]: total: 65536 MiB, used: 32768 MiB\nGPU[1]: total: 65536 MiB, used: 0 MiB\n"
            return "PID: 123\n\tHCU Index: ['0']\n"

        stdout = io.StringIO()
        with mock.patch("scripts.hygon_gpu_audit._rocm_smi_output", side_effect=output):
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    ["--select", "--allow-occupied", "--min-free-memory", "40"]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().splitlines(), ["0", "0,1", "0", "-"])


if __name__ == "__main__":
    unittest.main()
