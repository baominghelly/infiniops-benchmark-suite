#!/usr/bin/env python3
"""Run one platform workload from an immutable global InfiniBench manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any


DEFAULT_RUNS_ROOT = Path("/data-aisoft/infinibench_data/ops/.runs")
DEFAULT_OPS_ROOT = Path("/data-aisoft/infinibench_data/ops")
DEFAULT_WORKER_ROOT = Path("/data-aisoft/baoming/infinibench-workers")
DEFAULT_RUNNER = "/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh"
HEARTBEAT_INTERVAL_SECONDS = 30
HEARTBEAT_STALE_SECONDS = 120
ADAPTERS = {
    "nvidia-a100": "run-nvidia-a100-benchmark.sh",
    "metax-c550": "run-metax-c550-weekly.sh",
}
PLATFORM_ENV = {
    "nvidia-a100": {},
    "metax-c550": {
        "CONTAINER": "baoming_infiniops_metax",
        "INFINIOPS_SOURCE_ROOT": "/workspace/InfiniOps",
        "INFINIRT_SOURCE_ROOT": "/workspace/InfiniRT",
        "INFINIOPS_BUILD_DIR": "/workspace/.cache/infiniops-build",
    },
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def host_name() -> str:
    if hasattr(os, "uname"):
        return os.uname().nodename
    return os.environ.get("COMPUTERNAME", os.name)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        temp_path = Path(handle.name)
    temp_path.replace(path)


def load_manifest(runs_root: Path, global_run_id: str) -> dict[str, Any]:
    manifest_path = runs_root / global_run_id / "global-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"global manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("state") not in {"ready", "running", "failed"}:
        raise RuntimeError(f"global run state is not runnable: {manifest.get('state')}")
    if global_run_id != manifest.get("global_run_id"):
        raise RuntimeError("global manifest ID mismatch")
    return manifest


def output_has_entries(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def checkpoint_result_files(output_root: Path) -> list[Path]:
    """Return benchmark checkpoints that must never be silently replaced."""
    if not output_root.is_dir():
        return []
    return sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name in {"results.json", "results.jsonl"}
    )


def console_tail(text: str, limit: int = 12_000) -> str:
    return text[-limit:]


def classify_failure(
    console: Path,
    benchmark_output: Path,
    exit_code: int | None,
) -> dict[str, Any]:
    """Classify a completed worker failure and emit a conservative retry policy.

    Auto-retry is allowed only for known transient conditions and only when the
    failed run left no complete/checkpoint result files. Any partial result is
    preserved for manual review until shard-level resume is available.
    """
    try:
        text = console.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    lower = text.lower()
    result_files = checkpoint_result_files(benchmark_output)

    if (
        "torch.cuda.is_available()" in text
        and "assertionerror" in lower
    ) or "can't initialize nvml" in lower or "no cuda gpus are available" in lower:
        code = "container_cuda_lost"
    elif "cmake error" in lower or "ninja: build stopped" in lower:
        code = "build_failure"
    elif (
        "source preflight failed" in lower
        or "unable to access" in lower
        or "failed to connect" in lower
        or "connection timed out" in lower
        or "could not resolve host" in lower
    ):
        code = "source_network_failure"
    elif (
        "container not found" in lower
        or "container is not running" in lower
        or "docker socket is unavailable" in lower
        or "cannot connect to the docker daemon" in lower
    ):
        code = "benchmark_container_down"
    elif (
        "no idle" in lower
        or "has a running process" in lower
        or "utilization exceeds" in lower
        or "free memory is below" in lower
        or "foreign process" in lower
    ):
        code = "gpu_busy"
    elif "output directory already exists" in lower:
        code = "output_directory_nonempty"
    elif exit_code == 124 or "timed out" in lower:
        code = "command_timeout"
    elif exit_code in {130, 143}:
        code = "worker_stopped"
    else:
        code = "benchmark_failure"

    transient = code in {
        "container_cuda_lost",
        "source_network_failure",
        "benchmark_container_down",
        "gpu_busy",
        "command_timeout",
    }
    recommended = {
        "container_cuda_lost": "restart_benchmark_container_then_retry_without_rebuild",
        "source_network_failure": "retry_fixed_commit_after_source_network_recovery",
        "benchmark_container_down": "start_benchmark_container_then_retry",
        "gpu_busy": "wait_for_idle_accelerator_then_retry",
        "command_timeout": "inspect_hung_worker_before_any_retry",
        "build_failure": "inspect_build_log_and_rebuild",
        "output_directory_nonempty": "inspect_existing_output_before_explicit_reuse",
        "worker_stopped": "do_not_auto_retry_operator_stop",
        "benchmark_failure": "inspect_console_and_diagnostics_manually",
    }[code]

    has_checkpoint = bool(result_files)
    auto_retry = transient and not has_checkpoint
    return {
        "failure_code": code,
        "auto_retry": auto_retry,
        "max_auto_retries": 3 if auto_retry else 0,
        "retry_delay_seconds": 300 if auto_retry else None,
        "recommended_action": recommended,
        "preserve_partial_results": has_checkpoint,
        "allow_replace_existing_output": not has_checkpoint,
        "checkpoint_result_files": [str(path) for path in result_files],
        "checkpoint_result_count": len(result_files),
        "console_tail": console_tail(text),
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-run-id", required=True)
    parser.add_argument("--device", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--ops-root", type=Path, default=DEFAULT_OPS_ROOT)
    parser.add_argument("--worker-root", type=Path, default=DEFAULT_WORKER_ROOT)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--benchmark-runner", default=DEFAULT_RUNNER)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--replace-existing-output",
        action="store_true",
        help="explicitly allow the platform runner to reset this date directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.global_run_id):
        raise ValueError("invalid global run ID")
    manifest = load_manifest(args.runs_root, args.global_run_id)
    if args.device not in manifest.get("platforms", []):
        raise RuntimeError(f"{args.device} is not in global run platform list")

    execution_time = str(manifest["execution_time"])
    execution_date = execution_time[:8]
    benchmark_output = args.ops_root / args.device / execution_date
    adapter = args.adapter
    if adapter is None:
        release = (manifest.get("release") or {}).get("root")
        if not release:
            raise RuntimeError("manifest has no release.root; pass --adapter explicitly")
        adapter = Path(release) / "hermes" / ADAPTERS[args.device]
    adapter = adapter.resolve()
    if not adapter.is_file():
        raise RuntimeError(f"adapter script not found: {adapter}")

    state_root = args.worker_root / args.device / args.global_run_id
    console = state_root / "benchmark-console.log"
    worker_manifest = state_root / "worker-manifest.json"
    lock_path = state_root / ".worker.lock"
    source = manifest["source"]
    benchmark = manifest["benchmark"]

    base_payload: dict[str, Any] = {
        "schema_version": 1,
        "global_run_id": args.global_run_id,
        "device": args.device,
        "state": "pending",
        "host": host_name(),
        "pid": None,
        "adapter": str(adapter),
        "benchmark_runner": args.benchmark_runner,
        "benchmark_output": str(benchmark_output),
        "execution_time": execution_time,
        "infiniops_commit": source["infiniops_commit"],
        "infinirt_commit": source["infinirt_commit"],
        "dtypes": benchmark["dtypes"],
        "profiles": benchmark["profiles"],
        "attempts": 0,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "heartbeat_interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
        "heartbeat_stale_seconds": HEARTBEAT_STALE_SECONDS,
    }

    command = ["bash", str(adapter)]
    environment = os.environ.copy()
    environment.update(
        {
            "BENCHMARK_RUNNER": args.benchmark_runner,
            "EXECUTION_TIME": execution_time,
            "INFINIOPS_REF": source["infiniops_commit"],
            "INFINIRT_REF": source["infinirt_commit"],
            "DTYPES": ",".join(benchmark["dtypes"]),
            "PROFILES": ",".join(benchmark["profiles"]),
            "BUILD_INFINIOPS": environment.get("BUILD_INFINIOPS", "1"),
            "UPDATE_SOURCES": environment.get("UPDATE_SOURCES", "1"),
            "REUSE_RUN_DIR": "1" if args.replace_existing_output else "0",
        }
    )
    environment.update(PLATFORM_ENV[args.device])

    if args.dry_run:
        print(f"dry_run=true device={args.device}")
        print(f"global_manifest={args.runs_root / args.global_run_id / 'global-manifest.json'}")
        print(f"adapter={adapter}")
        print(f"benchmark_output={benchmark_output}")
        print(f"worker_manifest={worker_manifest}")
        print(f"infiniops_commit={source['infiniops_commit']}")
        print(f"infinirt_commit={source['infinirt_commit']}")
        print(f"replace_existing_output={args.replace_existing_output}")
        print(f"command={' '.join(command)}")
        return 0

    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - Windows is dry-run only.
        raise RuntimeError("file locking requires Linux; use --dry-run on Windows") from exc

    if output_has_entries(benchmark_output) and not args.replace_existing_output:
        raise RuntimeError(
            f"benchmark output already exists and is non-empty: {benchmark_output}; "
            "pass --replace-existing-output only if it is disposable"
        )

    state_root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another {args.device} worker is active: {lock_path}") from exc

        if worker_manifest.exists():
            previous = json.loads(worker_manifest.read_text(encoding="utf-8"))
            if previous.get("state") == "succeeded":
                raise RuntimeError("worker already succeeded; refusing to rerun")
            base_payload["attempts"] = int(previous.get("attempts", 0)) + 1
        else:
            base_payload["attempts"] = 1

        base_payload.update(
            {
                "state": "running",
                "pid": os.getpid(),
                "started_at": utc_now(),
                "heartbeat_at": utc_now(),
                "console": str(console),
            }
        )
        atomic_write_json(worker_manifest, base_payload)
        console.write_text("", encoding="utf-8")

        print(f"worker_start global_run_id={args.global_run_id} device={args.device}")
        heartbeat_stop = threading.Event()

        def write_heartbeat() -> None:
            while not heartbeat_stop.wait(HEARTBEAT_INTERVAL_SECONDS):
                snapshot = dict(base_payload)
                snapshot["heartbeat_at"] = utc_now()
                snapshot["updated_at"] = snapshot["heartbeat_at"]
                try:
                    atomic_write_json(worker_manifest, snapshot)
                except Exception as exc:
                    print(f"worker_heartbeat_write_failed error={exc!r}", flush=True)
                    return

        heartbeat_thread = threading.Thread(
            target=write_heartbeat,
            name="infinibench-worker-heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()

        process: subprocess.CompletedProcess[bytes] | None = None
        run_exception: BaseException | None = None
        try:
            with console.open("ab") as output_handle:
                process = subprocess.run(
                    command,
                    cwd=str(state_root),
                    env=environment,
                    stdout=output_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        except BaseException as exc:
            run_exception = exc
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=5)

        if process is None:
            return_code = 130 if isinstance(run_exception, KeyboardInterrupt) else 1
        else:
            return_code = process.returncode
        final_state = "succeeded" if process.returncode == 0 else "failed"
        final_state = "succeeded" if return_code == 0 else "failed"
        failure_policy = None
        if final_state == "failed":
            failure_policy = classify_failure(console, benchmark_output, return_code)

        ended_at = utc_now()
        base_payload.update(
            {
                "state": final_state,
                "exit_code": return_code,
                "started_at": base_payload["started_at"],
                "ended_at": ended_at,
                "updated_at": ended_at,
                "heartbeat_at": ended_at,
                "pid": None,
                "failure_policy": failure_policy,
                "failure_code": failure_policy["failure_code"] if failure_policy else None,
                "auto_retry": failure_policy["auto_retry"] if failure_policy else None,
            }
        )
        atomic_write_json(worker_manifest, base_payload)
        print(console.read_text(encoding="utf-8", errors="replace"), end="")
        print(
            f"worker_end global_run_id={args.global_run_id} "
            f"device={args.device} state={final_state} exit_code={return_code}"
        )
        if run_exception is not None:
            raise run_exception
        return return_code


if __name__ == "__main__":
    raise SystemExit(main())
