#!/usr/bin/env python3
"""Deterministic known-failure doctor for the NVIDIA A100 Hermes workflow."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


DEFAULT_RUNS_ROOT = Path("/data-aisoft/baoming/hermes/runs")
DEFAULT_CONTAINER = "baoming_infiniops"
DEFAULT_WORKFLOW = Path("/opt/data/scripts/run-nvidia-a100-benchmark.sh")
DEFAULT_OUTPUT_ROOT = Path("/data-aisoft/infinibench_data/ops/nvidia-a100")
RUN_NAME_RE = re.compile(r"(?:backfill-\d{8}-)?\d{8}")
TERMINAL_STATUSES = {
    "no_failure",
    "blocked",
    "retry_scheduled",
    "recovery_failed",
    "llm_triage_requested",
}


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def find_latest_run(runs_root: Path) -> Path | None:
    if not runs_root.is_dir():
        return None
    runs = [
        path
        for path in runs_root.iterdir()
        if path.is_dir() and RUN_NAME_RE.fullmatch(path.name)
    ]
    return max(runs, key=lambda path: path.name) if runs else None


def _combined_logs(run_root: Path, failure: Mapping[str, Any]) -> str:
    chunks: list[str] = []
    console = str(failure.get("final_console", ""))
    candidates: list[Path] = []
    if console:
        candidates.append(Path(console) if Path(console).is_absolute() else run_root / console)
    candidates.extend(sorted(run_root.glob("attempt-*.log")))
    for path in candidates:
        if path.is_file():
            try:
                chunks.append(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
    return "\n".join(chunks)


def classify_failure(
    run_root: Path, context: Mapping[str, Any]
) -> dict[str, Any]:
    failure = read_json(run_root / "failure.json")
    if not failure:
        return {
            "status": "no_failure",
            "failure_code": "none",
            "action": "none",
            "needs_llm": False,
        }

    text = _combined_logs(run_root, failure).lower()
    exit_code = int(failure.get("final_exit_code", -1) or -1)
    free_bytes = int(context.get("free_bytes", 0) or 0)
    free_gb = free_bytes / (1024**3)

    def result(code: str, action: str, needs_llm: bool) -> dict[str, Any]:
        return {
            "status": "classified",
            "failure_code": code,
            "action": action,
            "needs_llm": needs_llm,
            "final_exit_code": exit_code,
        }

    if "repository has local changes" in text or "refusing to overwrite" in text:
        return result("source_dirty", "capture_source_dirty_state", False)
    if context.get("docker_available", True) is False:
        return result("docker_down", "human_review", False)
    if context.get("container_running", True) is False:
        return result(
            "benchmark_container_down",
            "docker_start_benchmark_container",
            False,
        )
    if "no space left on device" in text or (
        context.get("disk_full_threshold_gb") is not None
        and free_gb <= float(context["disk_full_threshold_gb"])
    ):
        return result("disk_full", "prune_hermes_run_logs", False)
    if "output directory already exists and is non-empty" in text:
        return result("output_dir_nonempty", "schedule_retry", False)

    cuda_evidence = (
        "no cuda gpus are available" in text
        or (
            "assertionerror" in text
            and context.get("host_gpu_ok", False)
            and not context.get("container_cuda_ok", True)
        )
    )
    cuda_lost_probe = (
        context.get("host_gpu_ok", False)
        and not context.get("container_cuda_ok", True)
        and context.get("container_running", False)
    )
    if cuda_evidence or cuda_lost_probe:
        return result("container_cuda_lost", "docker_restart_benchmark_container", False)

    if (
        "process_owner=foreign" in text
        or "has a running process" in text
        or context.get("foreign_process_seen", False)
    ):
        return result("foreign_process", "schedule_retry", False)
    if "no idle nvidia gpu" in text or "no gpu satisfying the resource gate" in text:
        return result("no_idle_gpu", "schedule_retry", False)

    return result("unknown", "llm_triage", True)


def run_command(
    command: list[str], *, timeout: int = 20
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def collect_context(
    runs_root: Path,
    container: str,
    disk_full_threshold_gb: float = 10.0,
) -> dict[str, Any]:
    usage = shutil.disk_usage(runs_root if runs_root.exists() else Path("/"))
    docker_info = run_command(["docker", "info"])
    state = run_command(
        ["docker", "inspect", "-f", "{{.State.Running}}", container]
    )
    container_running = state.returncode == 0 and state.stdout.strip() == "true"
    # Hermes deliberately runs without GPU devices. Probe NVIDIA health through
    # the benchmark container over the Docker socket instead.
    accelerator_smi = (
        run_command(
            [
                "docker",
                "exec",
                container,
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader",
            ]
        )
        if container_running
        else subprocess.CompletedProcess(["nvidia-smi"], 1, "", "")
    )
    cuda = (
        run_command(
            [
                "docker",
                "exec",
                container,
                "python3",
                "-c",
                "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)",
            ]
        )
        if container_running
        else subprocess.CompletedProcess(["docker"], 1, "", "")
    )
    processes = (
        run_command(["docker", "top", container, "-eo", "pid,args"])
        if container_running
        else subprocess.CompletedProcess(["docker"], 1, "", "")
    )
    benchmark_running = bool(
        re.search(
            r"full_operator_benchmark|run_a100_benchmark|export_full_operator_report",
            processes.stdout,
            re.IGNORECASE,
        )
    )
    return {
        "docker_available": docker_info.returncode == 0,
        "host_gpu_ok": accelerator_smi.returncode == 0,
        "container_running": container_running,
        "container_cuda_ok": cuda.returncode == 0,
        "benchmark_process_running": benchmark_running,
        "free_bytes": usage.free,
        "disk_full_threshold_gb": disk_full_threshold_gb,
    }


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def prune_old_hermes_logs(
    runs_root: Path,
    current_run: Path,
    retain_days: int,
    max_bytes: int,
    now_epoch: int | None = None,
) -> dict[str, int]:
    """Prune only old files beneath runs_root, never the current run."""
    if not runs_root.is_dir() or not max_bytes or max_bytes <= 0:
        return {"files_removed": 0, "bytes_removed": 0, "bytes_skipped_by_cap": 0}

    now = time.time() if now_epoch is None else now_epoch
    cutoff = now - max(retain_days, 0) * 86400
    removed_files = 0
    removed_bytes = 0
    skipped = 0
    candidates: list[tuple[float, int, Path]] = []
    for path in runs_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_mtime >= cutoff or _is_under(path, current_run):
            continue
        if not _is_under(path, runs_root):
            continue
        candidates.append((stat.st_mtime, stat.st_size, path))

    for _, size, path in sorted(candidates):
        if removed_bytes + size > max_bytes:
            skipped += 1
            continue
        try:
            path.unlink()
        except OSError:
            continue
        removed_files += 1
        removed_bytes += size
    return {
        "files_removed": removed_files,
        "bytes_removed": removed_bytes,
        "bytes_skipped_by_cap": skipped,
    }


def _append_action(run_root: Path, action: Mapping[str, Any]) -> None:
    path = run_root / "doctor-actions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(action), ensure_ascii=False, sort_keys=True) + "\n")


def _signature(run_root: Path, failure: Mapping[str, Any]) -> str:
    payload = json.dumps(failure, sort_keys=True).encode("utf-8")
    for path in sorted(run_root.glob("attempt-*.log")):
        try:
            payload += path.read_bytes()
        except OSError:
            pass
    return hashlib.sha256(payload).hexdigest()


def _write_retry_request(
    run_root: Path,
    delay_seconds: int,
    environment: Mapping[str, str],
) -> None:
    write_json_atomic(
        run_root / "retry-request.json",
        {
            "not_before_epoch": int(time.time()) + delay_seconds,
            "environment": dict(environment),
            "reason": "deterministic-doctor-retry",
        },
    )


def apply_playbook(
    run_root: Path,
    classification: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    apply: bool,
    container: str,
    runs_root: Path,
) -> dict[str, Any]:
    code = str(classification["failure_code"])
    action = str(classification["action"])
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    needs_llm = bool(classification.get("needs_llm", False))
    status = "blocked" if code != "unknown" else "llm_triage_requested"
    details: dict[str, Any] = {}

    if code == "none":
        _append_action(
            run_root,
            {
                "timestamp": timestamp,
                "failure_code": code,
                "action": action,
                "apply": apply,
                "status": "no_failure",
                "details": {},
            },
        )
        return {
            "status": "no_failure",
            "failure_code": code,
            "action": action,
            "needs_llm": False,
            "details": details,
        }

    if code == "source_dirty":
        if apply:
            status_text = run_command(
                [
                    "docker",
                    "exec",
                    container,
                    "git",
                    "-c",
                    f"safe.directory=/workspace/InfiniOps",
                    "-C",
                    "/workspace/InfiniOps",
                    "status",
                    "--porcelain=v1",
                ]
            )
            diff_text = run_command(
                [
                    "docker",
                    "exec",
                    container,
                    "git",
                    "-c",
                    "safe.directory=/workspace/InfiniOps",
                    "-C",
                    "/workspace/InfiniOps",
                    "diff",
                    "--binary",
                ],
                timeout=60,
            )
            (run_root / "source-dirty-status.txt").write_text(
                status_text.stdout, encoding="utf-8", errors="replace"
            )
            (run_root / "source-dirty.diff").write_text(
                diff_text.stdout, encoding="utf-8", errors="replace"
            )
            details = {"human_review_required": True}
        else:
            status = "dry_run"

    elif code == "benchmark_container_down":
        guards_pass = context.get("docker_available", False)
        details = {"guards_pass": guards_pass}
        if not guards_pass:
            status = "blocked"
        elif apply:
            started = run_command(["docker", "start", container], timeout=120)
            state_after = run_command(
                ["docker", "inspect", "-f", "{{.State.Running}}", container]
            )
            running_after = state_after.stdout.strip() == "true"
            details = {
                "start_exit_code": started.returncode,
                "running_after": running_after,
            }
            if started.returncode == 0 and running_after:
                status = "retry_scheduled"
                _write_retry_request(
                    run_root,
                    30,
                    {
                        "REUSE_RUN_DIR": "1",
                        "UPDATE_SOURCES": "0",
                        "BUILD_INFINIOPS": "0",
                    },
                )
            else:
                status = "recovery_failed"
                needs_llm = True
        else:
            status = "dry_run"

    elif code == "docker_down":
        details = {"human_review_required": True}

    elif code == "disk_full":
        if apply:
            result = prune_old_hermes_logs(
                runs_root=runs_root,
                current_run=run_root,
                retain_days=30,
                max_bytes=10 * 1024**3,
            )
            details = result
            status = "retry_scheduled" if result["files_removed"] else "blocked"
            if result["files_removed"]:
                _write_retry_request(
                    run_root,
                    60,
                    {
                        "REUSE_RUN_DIR": "1",
                        "UPDATE_SOURCES": "0",
                        "BUILD_INFINIOPS": "1",
                    },
                )
        else:
            details = {"would_prune": True}
            status = "dry_run"

    elif code == "container_cuda_lost":
        guards_pass = (
            context.get("host_gpu_ok", False)
            and context.get("container_running", False)
            and not context.get("container_cuda_ok", True)
            and not context.get("benchmark_process_running", True)
        )
        details = {"guards_pass": guards_pass}
        if not guards_pass:
            needs_llm = True
            status = "recovery_failed"
        elif apply:
            restart = run_command(["docker", "restart", container], timeout=120)
            verify = run_command(
                [
                    "docker",
                    "exec",
                    container,
                    "python3",
                    "-c",
                    "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)",
                ]
            )
            details = {
                "restart_exit_code": restart.returncode,
                "verify_exit_code": verify.returncode,
                "verify_output": verify.stdout[-2000:],
            }
            if restart.returncode == 0 and verify.returncode == 0:
                status = "retry_scheduled"
                _write_retry_request(
                    run_root,
                    30,
                    {
                        "REUSE_RUN_DIR": "1",
                        "UPDATE_SOURCES": "0",
                        "BUILD_INFINIOPS": "0",
                    },
                )
            else:
                status = "recovery_failed"
                needs_llm = True
        else:
            status = "dry_run"

    elif code in {"foreign_process", "no_idle_gpu", "output_dir_nonempty"}:
        delay = 1800 if code in {"foreign_process", "no_idle_gpu"} else 60
        details = {
            "kill_foreign_process": False,
            "latency_comparison_allowed": False,
        }
        if apply:
            _write_retry_request(
                run_root,
                delay,
                {
                    "REUSE_RUN_DIR": "1",
                    "UPDATE_SOURCES": "0",
                    "BUILD_INFINIOPS": "1",
                },
            )
            status = "retry_scheduled"
        else:
            status = "dry_run"

    _append_action(
        run_root,
        {
            "timestamp": timestamp,
            "failure_code": code,
            "action": action,
            "apply": apply,
            "status": status,
            "details": details,
        },
    )
    return {
        "status": status,
        "failure_code": code,
        "action": action,
        "needs_llm": needs_llm,
        "details": details,
    }


def execute_due_retry(
    run_root: Path,
    workflow: Path,
    *,
    container: str,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    request = read_json(run_root / "retry-request.json")
    if not request:
        return {"status": "no_retry"}
    if int(time.time()) < int(request.get("not_before_epoch", 0)):
        return {"status": "retry_not_due"}
    if context.get("benchmark_process_running", False):
        return {"status": "benchmark_already_running"}

    environment = os.environ.copy()
    environment.update({str(k): str(v) for k, v in request.get("environment", {}).items()})
    log_path = run_root / (
        f"doctor-retry-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    )
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [str(workflow)],
                cwd=str(workflow.parent),
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
    except OSError as exc:
        return {"status": "retry_failed", "error": str(exc)}
    (run_root / "retry-request.json").unlink(missing_ok=True)
    return {
        "status": "retry_submitted",
        "exit_code": completed.returncode,
        "log": str(log_path),
    }


def doctor_once(
    runs_root: Path,
    *,
    apply: bool,
    container: str,
    workflow: Path,
    execute_retry: bool,
) -> dict[str, Any]:
    run_root = find_latest_run(runs_root)
    if run_root is None:
        state = {
            "status": "no_failure",
            "failure_code": "none",
            "needs_llm": False,
            "run_root": str(runs_root),
        }
        write_json_atomic(runs_root / "latest-doctor-state.json", state)
        return state

    context = collect_context(runs_root, container)
    failure = read_json(run_root / "failure.json")
    previous = read_json(run_root / "doctor-state.json")
    signature = _signature(run_root, failure)
    if previous and previous.get("signature") == signature and previous.get("status") in TERMINAL_STATUSES:
        write_json_atomic(runs_root / "latest-doctor-state.json", previous)
        if execute_retry:
            retry = execute_due_retry(
                run_root, workflow, container=container, context=context
            )
            previous["retry"] = retry
        return previous

    classification = classify_failure(run_root, context)
    result = apply_playbook(
        run_root,
        classification,
        context,
        apply=apply,
        container=container,
        runs_root=runs_root,
    )
    state = {
        **result,
        "signature": signature,
        "run_root": str(run_root),
        "execution_date": failure.get("execution_date", run_root.name),
        "benchmark_output_root": str(
            failure.get(
                "benchmark_output",
                "/data-aisoft/infinibench_data/ops/nvidia-a100",
            )
        ),
        "context": context,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    write_json_atomic(run_root / "doctor-state.json", state)
    write_json_atomic(runs_root / "latest-doctor-state.json", state)
    if execute_retry:
        retry = execute_due_retry(
            run_root, workflow, container=container, context=context
        )
        state["retry"] = retry
        write_json_atomic(run_root / "doctor-state.json", state)
        write_json_atomic(runs_root / "latest-doctor-state.json", state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--execute-retry", action="store_true")
    args = parser.parse_args(argv)

    try:
        import fcntl

        args.runs_root.mkdir(parents=True, exist_ok=True)
        lock_path = args.runs_root / ".doctor.lock"
        with lock_path.open("w", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            state = doctor_once(
                args.runs_root,
                apply=args.apply,
                container=args.container,
                workflow=args.workflow,
                execute_retry=args.execute_retry,
            )
    except ImportError:
        state = doctor_once(
            args.runs_root,
            apply=args.apply,
            container=args.container,
            workflow=args.workflow,
            execute_retry=args.execute_retry,
        )
    print(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
