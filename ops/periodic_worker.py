#!/usr/bin/env python3
"""Select and launch one eligible InfiniBench platform worker.

The launcher is intended to be invoked frequently by cron. It does not fork a
new worker when a healthy worker already has the per-run lease; a stale
heartbeat or an auto-retryable failed worker is resumed with the same immutable
global manifest.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_RUNS_ROOT = Path("/data-aisoft/infinibench_data/ops/.runs")
DEFAULT_CONTROL_ROOT = Path("/data-aisoft/infinibench_data/ops/.control")
DEFAULT_WORKER_ROOT = Path("/data-aisoft/infinibench_data/ops/.workers")
ADAPTERS = {
    "metax-c550": "run-metax-c550-weekly.sh",
    "nvidia-a100": "run-nvidia-a100-benchmark.sh",
}


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def parse_time(value: Any) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def heartbeat_is_fresh(worker: dict[str, Any]) -> bool:
    if worker.get("state") != "running":
        return False
    heartbeat = parse_time(worker.get("heartbeat_at"))
    if heartbeat is None:
        return False
    threshold = float(worker.get("heartbeat_stale_seconds", 120))
    return (utc_now() - heartbeat).total_seconds() <= threshold


def retry_delay_elapsed(worker: dict[str, Any]) -> bool:
    ended = parse_time(worker.get("ended_at"))
    if ended is None:
        return True
    policy = worker.get("failure_policy") or {}
    delay = float(policy.get("retry_delay_seconds") or 0)
    return (utc_now() - ended).total_seconds() >= delay


def select_candidate(
    runs_root: Path,
    control_root: Path,
    worker_root: Path,
    device: str,
    max_age_hours: float,
) -> tuple[str, str] | None:
    now = utc_now()
    manifests = sorted(runs_root.glob("*/global-manifest.json"), reverse=True)
    for manifest_path in manifests:
        manifest = read_json(manifest_path)
        if not manifest or device not in manifest.get("platforms", []):
            continue
        if manifest.get("state") == "canceled":
            continue

        run_id = str(manifest.get("global_run_id") or manifest_path.parent.name)
        if (manifest_path.parent / "control.json").is_file():
            control = read_json(manifest_path.parent / "control.json") or {}
            if control.get("command") == "cancel":
                continue
        if (control_root / "pause.json").is_file():
            return None

        created = parse_time((manifest.get("schedule") or {}).get("created_at"))
        if created and (now - created).total_seconds() > max_age_hours * 3600:
            continue

        worker_path = worker_root / device / run_id / "worker-manifest.json"
        worker = read_json(worker_path)
        if worker is None:
            return run_id, "worker state is missing"
        if worker.get("state") == "succeeded":
            continue
        if heartbeat_is_fresh(worker):
            continue

        attempts = max(1, int(worker.get("attempts") or 1))
        policy = worker.get("failure_policy") or {}
        max_attempts = max(1, int(policy.get("max_auto_retries", 3)))
        if attempts >= max_attempts:
            continue
        if worker.get("state") == "failed":
            auto_retry = bool(policy.get("auto_retry", worker.get("auto_retry", False)))
            if not auto_retry or attempts >= max_attempts:
                continue
            if not retry_delay_elapsed(worker):
                continue
            return run_id, f"auto retry {attempts}/{max_attempts}"

        if worker.get("state") == "running":
            return run_id, "worker heartbeat is stale"

    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=sorted(ADAPTERS), required=True)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--worker-root", type=Path, default=DEFAULT_WORKER_ROOT)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-age-hours", type=float, default=48.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = select_candidate(
        args.runs_root,
        args.control_root,
        args.worker_root,
        args.device,
        args.max_age_hours,
    )
    if selected is None:
        return 0

    run_id, reason = selected
    adapter = args.adapter or args.worker.with_name(ADAPTERS[args.device])
    command = [
        sys.executable,
        str(args.worker),
        "--global-run-id",
        run_id,
        "--device",
        args.device,
        "--worker-root",
        str(args.worker_root),
        "--adapter",
        str(adapter),
        "--resume-existing-output",
    ]
    print(
        f"periodic_launch device={args.device} global_run_id={run_id} reason={reason}",
        flush=True,
    )
    if args.dry_run:
        print("command=" + " ".join(command))
        return 0

    environment = os.environ.copy()
    os.execvpe(command[0], command, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
