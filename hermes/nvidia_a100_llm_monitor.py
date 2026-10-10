#!/usr/bin/env python3
"""Emit LLM triage input only when deterministic recovery cannot handle a run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_state(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def monitor_output(state_path: Path | None) -> str:
    """Return a stable no-op unless the latest Doctor state requests triage."""
    state = _read_state(state_path)
    if state is None or not state.get("needs_llm", False):
        return "NO_LLM_NEEDED"

    run_root = Path(state.get("run_root", ""))
    payload = {
        "needs_llm": True,
        "run_root": str(run_root),
        "failure_code": state.get("failure_code", "unknown"),
        "doctor_status": state.get("status", "unknown"),
        "doctor_actions": str(run_root / "doctor-actions.jsonl"),
        "failure_record": str(run_root / "failure.json"),
        "logs": [
            str(run_root / "attempt-1-exclusive.log"),
            str(run_root / "attempt-2-shared.log"),
            str(
                Path(
                    state.get(
                        "benchmark_output_root",
                        "/data-aisoft/infinibench_data/ops/nvidia-a100",
                    )
                )
                / state.get("execution_date", "")
                / "benchmark-launcher.log"
            ),
            str(
                Path(
                    state.get(
                        "benchmark_output_root",
                        "/data-aisoft/infinibench_data/ops/nvidia-a100",
                    )
                )
                / state.get("execution_date", "")
                / "benchmark-monitor.log"
            ),
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def main() -> int:
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("/data-aisoft/baoming/hermes/runs/latest-doctor-state.json"),
    )
    args = parser.parse_args()
    print(monitor_output(args.state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
