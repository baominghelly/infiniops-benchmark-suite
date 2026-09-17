#!/usr/bin/env python3
"""Select and audit Moore Threads GPUs for exclusive benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _command(argv: list[str]) -> str:
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=15)


def gpu_snapshot() -> list[dict[str, Any]]:
    payload = json.loads(_command(["mthreads-gmi", "-q", "--json"]))
    return payload.get("GPU", [])


def process_snapshot() -> list[dict[str, str]]:
    output = _command(["mthreads-gmi", "--process-monitor"])
    if "no running process" in output.lower():
        return []
    processes: list[dict[str, str]] = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 4 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        if fields[1] == "0":
            continue
        processes.append(
            {
                "gpu": fields[0],
                "pid": fields[1],
                "memory_util": fields[2],
                "gpu_util": fields[3],
            }
        )
    return processes


def _percent(value: str) -> int:
    return int(value.rstrip("%"))


def _used_mib(gpu: dict[str, Any]) -> int:
    value = gpu.get("FB Memory Usage", {}).get("Used", "0MiB")
    return int(str(value).removesuffix("MiB"))


def _total_mib(gpu: dict[str, Any]) -> int:
    value = gpu.get("FB Memory Usage", {}).get("Total", "0MiB")
    return int(str(value).removesuffix("MiB"))


def _free_memory_percent(gpu: dict[str, Any]) -> float:
    total = _total_mib(gpu)
    if total <= 0:
        raise RuntimeError("Moore GPU reported invalid total memory")
    return round((total - _used_mib(gpu)) * 100.0 / total, 2)


def _utilization(gpu: dict[str, Any]) -> int:
    return _percent(str(gpu.get("Utilization", {}).get("Gpu", "0%")))


def select_idle_gpu(
    samples: int,
    interval: float,
    max_utilization: int,
    min_free_memory: int,
    allow_occupied: bool,
) -> int:
    candidates: set[str] | None = None
    for sample in range(1, samples + 1):
        gpus = gpu_snapshot()
        processes = process_snapshot()
        occupied = {process["gpu"] for process in processes}
        current = {
            str(gpu.get("Index"))
            for gpu in gpus
            if _utilization(gpu) <= max_utilization
            and _free_memory_percent(gpu) >= min_free_memory
            and (allow_occupied or str(gpu.get("Index")) not in occupied)
        }
        if not gpus:
            raise RuntimeError("mthreads-gmi returned no GPUs")
        print(
            f"sample={sample} idle={','.join(sorted(current, key=int))} "
            f"processes={len(processes)}",
            flush=True,
        )
        candidates = current if candidates is None else candidates & current
        if sample != samples:
            time.sleep(interval)
    if not candidates:
        raise RuntimeError("no idle Moore GPU passed all samples")
    return min(candidates, key=int)


def assert_free(
    gpu_index: str,
    samples: int,
    interval: float,
    max_utilization: int,
    min_free_memory: int,
    allow_occupied: bool,
) -> None:
    for sample in range(1, samples + 1):
        gpus = {str(gpu.get("Index")): gpu for gpu in gpu_snapshot()}
        if gpu_index not in gpus:
            raise RuntimeError(f"Moore GPU {gpu_index} was not found")
        gpu = gpus[gpu_index]
        processes = process_snapshot()
        util = _utilization(gpu)
        free_memory = _free_memory_percent(gpu)
        occupied = any(process["gpu"] == gpu_index for process in processes)
        print(
            f"sample={sample} gpu={gpu_index} utilization={util}% "
            f"free_memory={free_memory:.2f}% processes={len(processes)} "
            f"occupied={str(occupied).lower()}",
            flush=True,
        )
        if util > max_utilization:
            raise RuntimeError(
                f"Moore GPU {gpu_index} utilization exceeds {max_utilization}%: {util}%"
            )
        if free_memory < min_free_memory:
            raise RuntimeError(
                f"Moore GPU {gpu_index} free memory is below {min_free_memory}%: "
                f"{free_memory:.2f}%"
            )
        if occupied and not allow_occupied:
            raise RuntimeError(f"Moore GPU {gpu_index} has an existing process")
        if sample != samples:
            time.sleep(interval)


def _cgroup(pid: str) -> str:
    try:
        return Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").replace("\n", " ")
    except OSError:
        return ""


def _cmdline(pid: str) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def watch(
    *,
    container_id: str,
    container: str,
    harness: str,
    done_file: Path,
    foreign_flag: Path,
    interval: float,
    allow_foreign: bool,
) -> int:
    consecutive_errors = 0
    while True:
        try:
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            processes = process_snapshot()
            consecutive_errors = 0
            if not processes:
                print(f"sample={timestamp} gpu_processes=none", flush=True)
            for process in processes:
                pid = process["pid"]
                cgroup = _cgroup(pid)
                cmdline = _cmdline(pid)
                if not cgroup:
                    owner = "exited_before_attribution"
                elif container_id in cgroup:
                    owner = "benchmark_container"
                else:
                    owner = "foreign_allowed" if allow_foreign else "foreign"
                    if not allow_foreign and (
                        not foreign_flag.exists() or foreign_flag.stat().st_size == 0
                    ):
                        foreign_flag.write_text(f"{pid}\\n", encoding="utf-8")
                    if not allow_foreign:
                        subprocess.run(
                            ["docker", "exec", container, "pkill", "-f", harness],
                            check=False,
                            timeout=15,
                        )
                print(
                    "sample={} pid={} gpu={} memUtil={} gpuUtil={} "
                    "process_owner={} cgroup={} cmdline={}".format(
                        timestamp, pid, process["gpu"], process["memory_util"],
                        process["gpu_util"], owner, cgroup, cmdline
                    ),
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 - transient GMI errors are retried
            consecutive_errors += 1
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            print(
                "sample={} monitor_error={} {}: {}".format(
                    timestamp, consecutive_errors, type(exc).__name__, exc
                ),
                flush=True,
            )
            if consecutive_errors >= 3:
                if not foreign_flag.exists() or foreign_flag.stat().st_size == 0:
                    foreign_flag.write_text(
                        f"monitor-error-{consecutive_errors}: {exc}\\n",
                        encoding="utf-8",
                    )
                subprocess.run(
                    ["docker", "exec", container, "pkill", "-f", harness],
                    check=False,
                    timeout=15,
                )

        if done_file.exists() and done_file.stat().st_size > 0:
            return 0
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="physical GPU index to audit")
    parser.add_argument("--select", action="store_true", help="print the first idle GPU index")
    parser.add_argument("--assert-free", action="store_true")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--max-utilization", type=int, default=5)
    parser.add_argument("--min-free-memory", type=int, default=0)
    parser.add_argument("--allow-occupied", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--container-id")
    parser.add_argument("--container")
    parser.add_argument("--harness")
    parser.add_argument("--done-file", type=Path)
    parser.add_argument("--foreign-flag", type=Path)
    args = parser.parse_args()

    if args.select:
        print(
            select_idle_gpu(
                args.samples,
                args.interval,
                args.max_utilization,
                args.min_free_memory,
                args.allow_occupied,
            )
        )
        return 0
    if args.assert_free:
        if not args.gpu:
            parser.error("--gpu is required with --assert-free")
        assert_free(
            args.gpu,
            args.samples,
            args.interval,
            args.max_utilization,
            args.min_free_memory,
            args.allow_occupied,
        )
        return 0
    if args.watch:
        required = (args.container_id, args.container, args.harness, args.done_file, args.foreign_flag)
        if any(value is None for value in required):
            parser.error("--watch requires --container-id, --container, --harness, --done-file, --foreign-flag")
        return watch(
            container_id=args.container_id,
            container=args.container,
            harness=args.harness,
            done_file=args.done_file,
            foreign_flag=args.foreign_flag,
            interval=args.interval,
            allow_foreign=args.allow_occupied,
        )
    parser.error("choose --select, --assert-free, or --watch")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGINT)
    except Exception as exc:  # noqa: BLE001 - CLI callers need actionable diagnostics
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
