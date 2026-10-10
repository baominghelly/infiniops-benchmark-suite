import json
import tempfile
import unittest
from pathlib import Path

from hermes.nvidia_a100_llm_monitor import monitor_output


class NvidiaA100LLMMonitorTests(unittest.TestCase):
    def test_stable_no_llm_output(self):
        self.assertEqual(monitor_output(None), "NO_LLM_NEEDED")

    def test_gate_only_emits_triage_when_doctor_requests_llm(self):
        with tempfile.TemporaryDirectory() as directory:
            state = {
                "needs_llm": False,
                "failure_code": "container_cuda_lost",
            }
            path = Path(directory) / "doctor-state.json"
            path.write_text(json.dumps(state), encoding="utf-8")
            self.assertEqual(monitor_output(path), "NO_LLM_NEEDED")

            state["needs_llm"] = True
            path.write_text(json.dumps(state), encoding="utf-8")
            output = monitor_output(path)
            payload = json.loads(output)
            self.assertTrue(payload["needs_llm"])
            self.assertEqual(payload["failure_code"], "container_cuda_lost")


if __name__ == "__main__":
    unittest.main()
