#!/usr/bin/env python3
"""Audit HCU processes reported by Hygon rocm-smi."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROCM_SMI = "/opt/dtk/bin/rocm-smi"
ROCM_SMI_LIBS = (
    "/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/lib:/opt/dtk/lib"
)


def _rocm_smi_output() -> str:
    env = os.environ.copy()
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ROCM_SMI_LIBS + (":" + existing if existing else "")
    result = subprocess.run(
        [ROCM_SMI, "--showpids"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"rocm-smi exited with {result.returncode}: {result.stdout}")
    return result.stdout


def _integers_in_brackets(value: str) -> list[str]:
    match = re.search(r"\[([^]]*)\]", value)
    if not match:
        return []
    return re.findall(r"\d+", match.group(1))


def processes() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in _rocm_smi_output().splitlines():
        if line.startswith("PID:"):
            if current is not None:
                records.append(current)
            match = re.match(r"PID:\s+(\d+)", line)
            if match is None:
                current = None
                continue
            current = {
                "pid": int(match.group(1)),
                "hcu_indices": [],
                "vram_used_mib": None,
            }
            continue
        if current is None:
            continue
        if line.lstrip().startswith("HCU Index:"):
            current["hcu_indices"] = _integers_in_brackets(line)
        elif line.lstrip().startswith("VRAM USED(MiB):"):
            match = re.match(r"\s*VRAM USED\(MiB\):\s+(\d+)", line)
            if match:
                current["vram_used_mib"] = int(match.group(1))
    if current is not None:
        records.append(current)
    return records


def _owner(pid: int, container_id: str) -> str:
    try:
        cgroup = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8")
    except OSError:
        return "exited_before_attribution"
    return "benchmark_container" if container_id in cgroup else "foreign"


def selected_processes(gpu: str, container_id: str | None = None) -> list[dict[str, Any]]:
    selected = []
    for record in processes():
        if gpu not in record["hcu_indices"]:
            continue
        item = dict(record)
        item["hcu_indices"] = [str(value) for value in record["hcu_indices"]]
        if container_id:
            item["process_owner"] = _owner(record["pid"], container_id)
        selected.append(item)
    return selected


def _write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True, help="physical HCU index")
    parser.add_argument("--container-id", help="full Docker container ID")
    parser.add_argument("--assert-free", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--done-file", type=Path)
    parser.add_argument("--foreign-flag", type=Path)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    container_id = args.container_id or ""
    records = selected_processes(args.gpu, container_id or None)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": args.gpu,
        "processes": records,
    }
    _write_json(args.json, payload)

    if args.assert_free:
        if records:
            print(json.dumps(payload, indent=2), file=sys.stderr)
            return 2
        return 0

    if not args.watch:
        print(json.dumps(payload, indent=2))
        return 0

    while args.done_file is None or not args.done_file.exists() or args.done_file.stat().st_size == 0:
        records = selected_processes(args.gpu, container_id or None)
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "gpu": args.gpu,
            "processes": records,
        }
        print(json.dumps(payload, separators=(",", ":")), flush=True)
        if any(item.get("process_owner") == "foreign" for item in records):
            if args.foreign_flag is not None:
                args.foreign_flag.parent.mkdir(parents=True, exist_ok=True)
                args.foreign_flag.write_text(
                    "\n".join(str(item["pid"]) for item in records if item.get("process_owner") == "foreign") + "\n",
                    encoding="utf-8",
                )
            return 1
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
