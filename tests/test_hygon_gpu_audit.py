import unittest
from unittest import mock

from scripts.hygon_gpu_audit import processes


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


if __name__ == "__main__":
    unittest.main()
