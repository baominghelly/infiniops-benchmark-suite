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


def _rocm_smi_output(arguments: list[str] | None = None) -> str:
    env = os.environ.copy()
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ROCM_SMI_LIBS + (":" + existing if existing else "")
    result = subprocess.run(
        [ROCM_SMI, *(arguments or ["--showpids"])],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"rocm-smi exited with {result.returncode}: {result.stdout}")
    return result.stdout


def device_indices() -> list[str]:
    output = _rocm_smi_output(["--showproductname"])
    indices = sorted(set(re.findall(r"(?:GPU|HCU)\[([0-9]+)\]", output)), key=int)
    if not indices:
        raise RuntimeError("rocm-smi product output did not contain HCU indices")
    return indices


def utilization_by_gpu() -> dict[str, int | None]:
    output = _rocm_smi_output(["--showuse"])
    values: dict[str, int | None] = {}
    for line in output.splitlines():
        match = re.match(
            r"\s*(?:GPU|HCU)\[([0-9]+)\]\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%", line
        )
        if match:
            values[match.group(1)] = int(float(match.group(2)))
    return values


def _size_to_mib(value: str, unit: str | None) -> float:
    units = {
        "B": 1.0 / (1024 * 1024),
        "KB": 1.0 / 1024,
        "KiB": 1.0 / 1024,
        "MB": 1.0,
        "MiB": 1.0,
        "GB": 1024.0,
        "GiB": 1024.0,
    }
    return float(value) * units.get(unit or "MiB", 1.0)


def _memory_from_output(output: str) -> dict[str, float]:
    values: dict[str, float] = {}
    blocks = re.split(r"(?m)(?=^\s*(?:GPU|HCU)\[[0-9]+\])", output)
    for block in blocks:
        header = re.match(r"\s*(?:GPU|HCU)\[([0-9]+)\]", block)
        if header is None:
            continue
        index = header.group(1)
        total_match = re.search(
            r"\b(?:vram[_ ])?total\b\D*([0-9]+(?:\.[0-9]+)?)\s*(GiB|MiB|KiB|MB|KB|B)?",
            block,
            flags=re.IGNORECASE,
        )
        used_match = re.search(
            r"\b(?:vram[_ ])?used\b\D*([0-9]+(?:\.[0-9]+)?)\s*(GiB|MiB|KiB|MB|KB|B)?",
            block,
            flags=re.IGNORECASE,
        )
        if total_match and used_match:
            total = _size_to_mib(total_match.group(1), total_match.group(2))
            used = _size_to_mib(used_match.group(1), used_match.group(2))
            if total > 0:
                values[index] = max(0.0, min(100.0, (total - used) * 100.0 / total))
                continue

        percentages = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*%", block)
        if percentages:
            free_match = re.search(
                r"\bfree\b\D*([0-9]+(?:\.[0-9]+)?)\s*%", block, flags=re.IGNORECASE
            )
            if free_match:
                used_percent = 100.0 - float(free_match.group(1))
            else:
                used_percent = float(percentages[-1])
            values[index] = max(0.0, min(100.0, 100.0 - used_percent))
    return values


def free_memory_by_gpu() -> dict[str, float]:
    try:
        detailed = _memory_from_output(_rocm_smi_output(["--showmeminfo", "vram"]))
    except RuntimeError:
        detailed = {}
    if detailed:
        return detailed
    return _memory_from_output(_rocm_smi_output(["--showmemuse"]))


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


def _stop_container_process(container: str, harness: str) -> None:
    subprocess.run(
        ["docker", "exec", container, "pkill", "-f", harness],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


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


def select_gpu(
    *,
    max_utilization: int,
    min_free_memory: float,
    allow_occupied: bool,
) -> tuple[str, str, str, str]:
    busy = {
        index
        for record in processes()
        for index in record["hcu_indices"]
    }
    utilization = utilization_by_gpu()
    free_memory = free_memory_by_gpu()

    eligible = []
    rejected = []
    for index in device_indices():
        reasons = []
        if index in busy and not allow_occupied:
            reasons.append("busy")
        utilization_value = utilization.get(index)
        if utilization_value is None:
            reasons.append("invalid_utilization=None")
        elif utilization_value > max_utilization:
            reasons.append(f"utilization={utilization_value}")

        free_memory_value = free_memory.get(index)
        if free_memory_value is None:
            reasons.append("invalid_memory=None")
        elif free_memory_value < min_free_memory:
            reasons.append(f"free_memory={free_memory_value:.2f}%")

        if reasons:
            rejected.append(f"{index}({','.join(reasons)})")
        else:
            eligible.append(index)

    if not eligible:
        raise RuntimeError(
            "no Hygon HCU satisfies the resource gate; rejected: "
            + (", ".join(rejected) if rejected else "unknown")
        )
    return (
        eligible[0],
        ",".join(eligible),
        ",".join(sorted(busy, key=int)) or "-",
        ";".join(rejected) or "-",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="physical HCU index")
    parser.add_argument("--container-id", help="full Docker container ID")
    parser.add_argument("--container", help="benchmark Docker container name")
    parser.add_argument("--harness", help="benchmark harness path to stop on foreign process")
    parser.add_argument("--assert-free", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--max-utilization", type=int, default=5)
    parser.add_argument("--min-free-memory", type=float, default=0.0)
    parser.add_argument("--allow-occupied", action="store_true")
    parser.add_argument("--done-file", type=Path)
    parser.add_argument("--foreign-flag", type=Path)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    if args.select:
        selected, candidates, busy, rejected = select_gpu(
            max_utilization=args.max_utilization,
            min_free_memory=args.min_free_memory,
            allow_occupied=args.allow_occupied,
        )
        print("\n".join((selected, candidates, busy, rejected)))
        return 0

    if not args.gpu:
        raise SystemExit("--gpu is required unless --select is used")

    container_id = args.container_id or ""
    records = selected_processes(args.gpu, container_id or None)
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": args.gpu,
        "processes": records,
    }
    _write_json(args.json, payload)

    if args.assert_free:
        utilization = utilization_by_gpu().get(args.gpu)
        free_memory = free_memory_by_gpu().get(args.gpu)
        resource_failed = (
            (bool(records) and not args.allow_occupied)
            or utilization is None
            or utilization > args.max_utilization
            or free_memory is None
            or free_memory < args.min_free_memory
        )
        if resource_failed:
            print(json.dumps(payload, indent=2), file=sys.stderr)
            if utilization is not None:
                print(f"utilization={utilization}", file=sys.stderr)
            if free_memory is not None:
                print(f"free_memory_percent={free_memory:.2f}", file=sys.stderr)
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
        if args.allow_occupied:
            for item in records:
                if item.get("process_owner") == "foreign":
                    item["process_owner"] = "foreign_allowed"
        print(json.dumps(payload, separators=(",", ":")), flush=True)
        if any(item.get("process_owner") == "foreign" for item in records):
            if args.foreign_flag is not None:
                args.foreign_flag.parent.mkdir(parents=True, exist_ok=True)
                args.foreign_flag.write_text(
                    "\n".join(str(item["pid"]) for item in records if item.get("process_owner") == "foreign") + "\n",
                    encoding="utf-8",
                )
            if args.container and args.harness:
                _stop_container_process(args.container, args.harness)
            return 1
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
