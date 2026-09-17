#!/usr/bin/env python3
"""Audit Cambricon MLU utilization and process ownership with cnmon."""

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


CNMON = os.environ.get("CNMON_COMMAND", "cnmon")
_CARD_RE = re.compile(
    r"^\|\s*(\d+)\s+/\s+(\S+)\s+\S+\s*\|[^|]+\|\s*(\d+)%"
)
_MEMORY_RE = re.compile(
    r"^\|\s*\d+\s+\d+C\s+.*?\|\s*(\d+)\s+MiB/\s*(\d+)\s+MiB\s*\|"
)
_PROCESS_RE = re.compile(
    r"^\|\s*(\d+)\s+(\S+)\s+(\d+)\s+(.*?)\s+(\d+)\s+MiB\s*\|$"
)


def _cnmon_output() -> str:
    result = subprocess.run(
        [CNMON, "all"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"cnmon all exited with {result.returncode}: {result.stdout}")
    return result.stdout


def _cnmon_memory_output() -> str:
    result = subprocess.run(
        [CNMON, "info", "-m"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cnmon info -m exited with {result.returncode}: {result.stdout}"
        )
    return result.stdout


def parse_snapshot(output: str) -> dict[str, Any]:
    devices: dict[str, dict[str, Any]] = {}
    processes: list[dict[str, Any]] = []
    current_device: str | None = None
    in_processes = False

    for line in output.splitlines():
        if not in_processes:
            match = _CARD_RE.match(line)
            if match:
                current_device = match.group(1)
                devices[current_device] = {
                    "device_id": int(current_device),
                    "name": match.group(2),
                    "utilization": int(match.group(3)),
                    "memory_used_mib": None,
                    "memory_total_mib": None,
                    "memory_reserved_mib": None,
                    "memory_free_mib": None,
                }
                continue
            match = _MEMORY_RE.match(line)
            if match and current_device is not None:
                devices[current_device]["memory_used_mib"] = int(match.group(1))
                devices[current_device]["memory_total_mib"] = int(match.group(2))
                current_device = None
            if "| Processes:" in line:
                in_processes = True
            continue

        match = _PROCESS_RE.match(line)
        if match:
            processes.append(
                {
                    "device_id": int(match.group(1)),
                    "module_id": match.group(2),
                    "pid": int(match.group(3)),
                    "command": match.group(4).strip(),
                    "memory_used_mib": int(match.group(5)),
                }
            )

    return {"devices": devices, "processes": processes}


def parse_memory_details(output: str) -> dict[str, dict[str, int]]:
    details: dict[str, dict[str, int]] = {}
    current_device: str | None = None
    in_physical_memory = False
    for line in output.splitlines():
        card_match = re.match(r"^Card\s+(\d+)\s*$", line)
        if card_match:
            current_device = card_match.group(1)
            details[current_device] = {}
            in_physical_memory = False
            continue
        if current_device is None:
            continue
        if line.strip() == "Physical Memory Usage":
            in_physical_memory = True
            continue
        if in_physical_memory:
            value_match = re.match(
                r"^\s*(Total|Used|Reserved|Free)\s*:\s*(\d+)\s+MiB\s*$",
                line,
            )
            if value_match:
                details[current_device][value_match.group(1).lower()] = int(
                    value_match.group(2)
                )
                if value_match.group(1) == "Free":
                    in_physical_memory = False
    return details


def snapshot() -> dict[str, Any]:
    payload = parse_snapshot(_cnmon_output())
    memory_details = parse_memory_details(_cnmon_memory_output())
    for device_id, device in payload["devices"].items():
        detail = memory_details.get(device_id, {})
        device["memory_reserved_mib"] = detail.get("reserved")
        device["memory_free_mib"] = detail.get("free")
    return payload


def _owner(pid: int, container_id: str) -> str:
    try:
        cgroup = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8")
    except OSError:
        return "exited_before_attribution"
    return "benchmark_container" if container_id in cgroup else "foreign"


def selected_processes(
    device: str, container_id: str | None = None
) -> list[dict[str, Any]]:
    payload = snapshot()
    records = []
    for record in payload["processes"]:
        if str(record["device_id"]) != str(device):
            continue
        item = dict(record)
        if container_id:
            item["process_owner"] = _owner(record["pid"], container_id)
        records.append(item)
    return records


def _selection(
    payload: dict[str, Any],
    max_utilization: int,
    min_free_memory_percent: int,
    allow_occupied: bool = False,
):
    busy = {
        str(record["device_id"])
        for record in payload["processes"]
    }
    candidates = []
    rejected = []
    for device_id, device in sorted(payload["devices"].items()):
        reasons = []
        memory_total = device.get("memory_total_mib")
        memory_free = device.get("memory_free_mib")
        if device_id in busy and not allow_occupied:
            reasons.append("busy")
        if device["utilization"] > max_utilization:
            reasons.append(f"utilization={device['utilization']}%")
        if memory_total is None or memory_free is None:
            reasons.append("memory_unknown")
        elif memory_total <= 0:
            reasons.append("invalid_memory_total")
        else:
            free_memory = memory_free * 100.0 / memory_total
            if free_memory < min_free_memory_percent:
                reasons.append(f"free_memory={free_memory:.2f}%")
        if reasons:
            rejected.append(f"{device_id}:{','.join(reasons)}")
        else:
            candidates.append(device_id)
    return candidates, sorted(busy), rejected


def select_idle_device(
    samples: int = 1,
    interval: float = 2.0,
    max_utilization: int = 5,
    min_free_memory_percent: int = 0,
    allow_occupied: bool = False,
)-> dict[str, Any]:
    candidates: set[str] | None = None
    observations = []
    for sample_index in range(samples):
        payload = snapshot()
        current_candidates, busy, rejected = _selection(
            payload, max_utilization, min_free_memory_percent, allow_occupied
        )
        observations.append(
            {
                "sample": sample_index + 1,
                "candidates": current_candidates,
                "busy": busy,
                "rejected": rejected,
            }
        )
        current_set = set(current_candidates)
        candidates = current_set if candidates is None else candidates & current_set
        if sample_index + 1 < samples:
            time.sleep(interval)

    final_candidates = sorted(candidates or [])
    observations_busy = sorted(
        {item for observation in observations for item in observation["busy"]}
    )
    observations_rejected = sorted(
        {
            item.rsplit(":", 1)[0]
            for observation in observations
            for item in observation["rejected"]
        }
    )
    return {
        "selected_mlu": int(final_candidates[0]) if final_candidates else None,
        "candidates": final_candidates,
        "busy": observations_busy,
        "rejected": observations_rejected,
        "observations": observations,
    }


def _json_payload(
    device: str,
    container_id: str | None,
    allow_occupied: bool = False,
) -> dict[str, Any]:
    payload = snapshot()
    device_record = payload["devices"].get(str(device))
    processes = []
    for record in payload["processes"]:
        if str(record["device_id"]) != str(device):
            continue
        item = dict(record)
        if container_id:
            owner = _owner(record["pid"], container_id)
            if owner == "foreign" and allow_occupied:
                owner = "foreign_allowed"
            item["process_owner"] = owner
        processes.append(item)
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mlu": str(device),
        "device": device_record,
        "processes": processes,
    }


def _write_foreign_flag(path: Path | None, records: list[dict[str, Any]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    foreign_pids = [str(item["pid"]) for item in records if item.get("process_owner") == "foreign"]
    path.write_text("\n".join(foreign_pids) + ("\n" if foreign_pids else ""), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlu", help="physical MLU index")
    parser.add_argument("--container-id", help="full Docker container ID")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--shell", action="store_true")
    parser.add_argument("--assert-free", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--done-file", type=Path)
    parser.add_argument("--foreign-flag", type=Path)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--max-utilization", type=int, default=5)
    parser.add_argument("--min-free-memory-percent", type=int, default=0)
    parser.add_argument("--allow-occupied", action="store_true")
    args = parser.parse_args(argv)

    if args.select:
        payload = select_idle_device(
            samples=args.samples,
            interval=args.interval,
            max_utilization=args.max_utilization,
            min_free_memory_percent=args.min_free_memory_percent,
            allow_occupied=args.allow_occupied,
        )
        if args.shell:
            selected = "" if payload["selected_mlu"] is None else payload["selected_mlu"]
            print(f"physical_mlu={selected}")
            print("gpu_selection=auto")
            print(f"gpu_candidates={','.join(payload['candidates'])}")
            print(f"gpu_busy={','.join(payload['busy'])}")
            print(f"gpu_rejected={','.join(payload['rejected'])}")
        else:
            print(json.dumps(payload, indent=2))
        return 0 if payload["selected_mlu"] is not None else 2

    if not args.mlu:
        parser.error("--mlu is required unless --select is used")
    if args.samples < 1 or args.interval <= 0:
        parser.error("--samples must be positive and --interval must be greater than zero")

    container_id = args.container_id or ""
    if args.assert_free:
        payload = snapshot()
        candidates, busy, rejected = _selection(
            payload,
            args.max_utilization,
            args.min_free_memory_percent,
            args.allow_occupied,
        )
        if str(args.mlu) not in candidates:
            print(
                json.dumps(
                    {
                        "mlu": args.mlu,
                        "busy": busy,
                        "rejected": rejected,
                        "processes": [
                            item
                            for item in payload["processes"]
                            if str(item["device_id"]) == str(args.mlu)
                        ],
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 2
        return 0

    if not args.watch:
        print(
            json.dumps(
                _json_payload(args.mlu, container_id or None, args.allow_occupied),
                indent=2,
            )
        )
        return 0

    while args.done_file is None or not args.done_file.exists() or args.done_file.stat().st_size == 0:
        payload = _json_payload(args.mlu, container_id or None, args.allow_occupied)
        print(json.dumps(payload, separators=(",", ":")), flush=True)
        foreign = [item for item in payload["processes"] if item.get("process_owner") == "foreign"]
        if foreign:
            _write_foreign_flag(args.foreign_flag, payload["processes"])
            return 1
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
