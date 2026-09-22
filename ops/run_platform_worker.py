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
from pathlib import Path
from typing import Any


DEFAULT_RUNS_ROOT = Path("/data-aisoft/infinibench_data/ops/.runs")
DEFAULT_OPS_ROOT = Path("/data-aisoft/infinibench_data/ops")
DEFAULT_WORKER_ROOT = Path("/data-aisoft/baoming/infinibench-workers")
DEFAULT_RUNNER = "/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh"
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
            "BUILD_INFINIOPS": "1",
            "UPDATE_SOURCES": "1",
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
                "console": str(console),
            }
        )
        atomic_write_json(worker_manifest, base_payload)
        console.write_text("", encoding="utf-8")

        print(f"worker_start global_run_id={args.global_run_id} device={args.device}")
        with console.open("ab") as output_handle:
            process = subprocess.run(
                command,
                cwd=str(state_root),
                env=environment,
                stdout=output_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )

        final_state = "succeeded" if process.returncode == 0 else "failed"
        base_payload.update(
            {
                "state": final_state,
                "exit_code": process.returncode,
                "started_at": base_payload["started_at"],
                "ended_at": utc_now(),
                "updated_at": utc_now(),
                "pid": None,
            }
        )
        atomic_write_json(worker_manifest, base_payload)
        print(console.read_text(encoding="utf-8", errors="replace"), end="")
        print(
            f"worker_end global_run_id={args.global_run_id} "
            f"device={args.device} state={final_state} exit_code={process.returncode}"
        )
        return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
