#!/usr/bin/env python3
"""Create an immutable InfiniBench global run manifest."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_RUNS_ROOT = Path("/data-aisoft/infinibench_data/ops/.runs")
DEFAULT_INFINIOPS_REPO = "https://github.com/InfiniTensor/InfiniOps.git"
DEFAULT_INFINIRT_REPO = "https://github.com/InfiniTensor/InfiniRT.git"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


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


def resolve_commit(repository: str, requested_ref: str) -> str:
    if COMMIT_RE.fullmatch(requested_ref):
        return requested_ref.lower()
    result = subprocess.run(
        ["git", "ls-remote", repository, requested_ref],
        check=False,
        text=True,
        capture_output=True,
        timeout=90,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"unable to resolve {requested_ref!r} in {repository}: "
            f"{result.stderr.strip() or result.returncode}"
        )
    for line in result.stdout.splitlines():
        commit = line.split()[0] if line.split() else ""
        if COMMIT_RE.fullmatch(commit):
            return commit.lower()
    raise RuntimeError(f"ref {requested_ref!r} did not resolve to a commit in {repository}")


def validate_execution_time(value: str) -> None:
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}", value):
        raise ValueError("EXECUTION_TIME must use YYYYMMDD-HHMMSS")
    try:
        dt.datetime.strptime(value, "%Y%m%d-%H%M%S")
    except ValueError as exc:
        raise ValueError(f"invalid EXECUTION_TIME: {value}") from exc


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--global-run-id")
    parser.add_argument("--execution-time")
    parser.add_argument("--schedule-mode", default="manual")
    parser.add_argument(
        "--platform",
        action="append",
        choices=["nvidia-a100", "metax-c550"],
        default=None,
        help="participating platform; repeat for multiple platforms",
    )
    parser.add_argument("--infiniops-repo", default=DEFAULT_INFINIOPS_REPO)
    parser.add_argument("--infinirt-repo", default=DEFAULT_INFINIRT_REPO)
    parser.add_argument("--infiniops-ref", default="master")
    parser.add_argument("--infinirt-ref", default="master")
    parser.add_argument("--infiniops-commit")
    parser.add_argument("--infinirt-commit")
    parser.add_argument("--dtypes", default="float16,float32,bfloat16")
    parser.add_argument("--profiles", default="decode-4k")
    parser.add_argument("--release-root")
    parser.add_argument("--release-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    execution_time = args.execution_time or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    validate_execution_time(execution_time)
    platforms = args.platform or ["nvidia-a100", "metax-c550"]
    global_run_id = args.global_run_id or f"{execution_time}-{args.schedule_mode}"
    if not RUN_ID_RE.fullmatch(global_run_id):
        raise ValueError("global run ID must be ASCII letters, digits, dot, underscore, or dash")

    infiniopts_commit = args.infiniops_commit or resolve_commit(
        args.infiniops_repo, args.infiniops_ref
    )
    infinirt_commit = args.infinirt_commit or resolve_commit(
        args.infinirt_repo, args.infinirt_ref
    )
    if not COMMIT_RE.fullmatch(infiniopts_commit) or not COMMIT_RE.fullmatch(infinirt_commit):
        raise ValueError("source commits must be full 40-character SHA-1 values")

    dtypes = [item.strip() for item in args.dtypes.split(",") if item.strip()]
    profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]
    if not dtypes or not profiles:
        raise ValueError("dtypes and profiles cannot be empty")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "global_run_id": global_run_id,
        "state": "ready",
        "schedule": {
            "mode": args.schedule_mode,
            "created_at": utc_now(),
        },
        "execution_time": execution_time,
        "source": {
            "infiniops_repo": args.infiniops_repo,
            "infinirt_repo": args.infinirt_repo,
            "infiniops_commit": infiniopts_commit.lower(),
            "infinirt_commit": infinirt_commit.lower(),
        },
        "benchmark": {
            "dtypes": dtypes,
            "profiles": profiles,
        },
        "platforms": platforms,
        "release": {
            "root": args.release_root,
            "sha256": args.release_sha256,
        },
    }

    run_root = args.runs_root / global_run_id
    manifest_path = run_root / "global-manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        immutable_fields = (
            "global_run_id",
            "execution_time",
            "source",
            "benchmark",
            "platforms",
            "release",
        )
        differences = [
            field
            for field in immutable_fields
            if existing.get(field) != payload.get(field)
        ]
        if differences:
            raise RuntimeError(f"global run already exists with different metadata: {manifest_path}")
        if existing.get("schedule", {}).get("mode") != payload["schedule"]["mode"]:
            raise RuntimeError(f"global run schedule mode differs: {manifest_path}")
        print(f"idempotent=true manifest={manifest_path}")
        return 0
    atomic_write_json(manifest_path, payload)
    print(f"created=true manifest={manifest_path}")
    print(f"infiniops_commit={infiniopts_commit.lower()}")
    print(f"infinirt_commit={infinirt_commit.lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
