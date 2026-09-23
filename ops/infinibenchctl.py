#!/usr/bin/env python3
"""Operational control entry for the two-platform InfiniBench pilot."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any


OPS_ROOT = Path("/data-aisoft/infinibench_data/ops")
RUNS_ROOT = OPS_ROOT / ".runs"
CONTROL_ROOT = OPS_ROOT / ".control"
WORKER_ROOT = OPS_ROOT / ".workers"
LAUNCH_ROOT = Path("/data-aisoft/baoming/infinibench-ops/launches")
RELEASE_ROOT = "/data-aisoft/infinibench-releases/infinibench-ops-27c5038-v1"
RELEASE_SHA256 = "37e0f533ba69957145c55eb9c3143990f8360ba4dda955fd14fd0b616d793726"
HERMES = "baoming_hermes_agent"
NVIDIA_CONTAINER = "baoming_infiniops"
METAX_SSH = "metax"
METAX_CONTAINER = "baoming_infiniops_metax"
DEVICES = ("nvidia-a100", "metax-c550")


def beijing_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))


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
        temporary = Path(handle.name)
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pause_marker() -> Path:
    return CONTROL_ROOT / "pause.json"


def is_paused() -> bool:
    return pause_marker().is_file()


def run_command(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        text=True,
        capture_output=capture,
        encoding="utf-8",
        errors="replace",
    )


def worker_manifest_path(device: str, global_run_id: str) -> Path:
    return device_worker_root(device) / device / global_run_id / "worker-manifest.json"


def device_worker_root(device: str) -> Path:
    if device == "nvidia-a100":
        return Path("/data-aisoft/baoming/hermes/runs/global-workers")
    return WORKER_ROOT


def safe_run_id(value: str) -> None:
    import re

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise SystemExit(f"unsafe global run id: {value}")


def create_manifest(args: argparse.Namespace) -> None:
    manifest_path = (
        RUNS_ROOT / args.global_run_id / "global-manifest.json"
    )
    if manifest_path.exists():
        existing = read_json(manifest_path)
        expected = {
            "global_run_id": args.global_run_id,
            "execution_time": args.execution_time,
            "platforms": args.device,
        }
        for key, value in expected.items():
            if existing.get(key) != value:
                raise RuntimeError(f"existing global run has different {key}: {manifest_path}")
        if existing.get("schedule", {}).get("mode") != args.schedule_mode:
            raise RuntimeError(f"existing global run has different schedule mode: {manifest_path}")
        if existing.get("benchmark", {}).get("dtypes") != [
            item.strip() for item in args.dtypes.split(",") if item.strip()
        ]:
            raise RuntimeError(f"existing global run has different dtypes: {manifest_path}")
        if existing.get("benchmark", {}).get("profiles") != [
            item.strip() for item in args.profile.split(",") if item.strip()
        ]:
            raise RuntimeError(f"existing global run has different profiles: {manifest_path}")
        if args.infiniops_commit and existing.get("source", {}).get("infiniops_commit") != args.infiniops_commit:
            raise RuntimeError(f"existing global run has different InfiniOps commit: {manifest_path}")
        if args.infinirt_commit and existing.get("source", {}).get("infinirt_commit") != args.infinirt_commit:
            raise RuntimeError(f"existing global run has different InfiniRT commit: {manifest_path}")
        print(f"idempotent=true manifest={manifest_path}")
        print(f"infiniops_commit={existing['source']['infiniops_commit']}")
        print(f"infinirt_commit={existing['source']['infinirt_commit']}")
        return
    common = [
        "--global-run-id",
        args.global_run_id,
        "--execution-time",
        args.execution_time,
        "--schedule-mode",
        args.schedule_mode,
        "--dtypes",
        args.dtypes,
        "--profiles",
        args.profile,
        "--release-root",
        RELEASE_ROOT,
        "--release-sha256",
        RELEASE_SHA256,
    ]
    for device in args.device:
        common += ["--platform", device]

    direct = "https://github.com/InfiniTensor"
    proxy = "https://gh-proxy.com/https://github.com/InfiniTensor"
    candidates = [(f"{direct}/InfiniOps.git", f"{direct}/InfiniRT.git")]
    if not args.infiniops_commit or not args.infinirt_commit:
        candidates.append((f"{proxy}/InfiniOps.git", f"{proxy}/InfiniRT.git"))

    last_output = ""
    for index, (infiniops_repo, infinirt_repo) in enumerate(candidates, start=1):
        command = [
            "docker",
            "exec",
            HERMES,
            "python3",
            "/opt/data/scripts/create_global_run.py",
            *common,
            "--infiniops-repo",
            infiniops_repo,
            "--infinirt-repo",
            infinirt_repo,
        ]
        if args.infiniops_commit:
            command += ["--infiniops-commit", args.infiniops_commit]
        else:
            command += ["--infiniops-ref", args.infiniops_ref]
        if args.infinirt_commit:
            command += ["--infinirt-commit", args.infinirt_commit]
        else:
            command += ["--infinirt-ref", args.infinirt_ref]
        completed = run_command(command, capture=True)
        last_output = ((completed.stdout or "") + (completed.stderr or "")).strip()
        if completed.returncode == 0:
            print(last_output)
            return
        print(f"manifest_candidate={index} failed rc={completed.returncode}", flush=True)
    raise SystemExit(f"unable to create global manifest:\n{last_output}")


def launch_device(
    device: str,
    global_run_id: str,
    *,
    replace_existing_output: bool,
) -> dict[str, Any]:
    common = [
        "--global-run-id",
        global_run_id,
        "--device",
        device,
    ]
    if replace_existing_output:
        common.append("--replace-existing-output")

    if device == "nvidia-a100":
        inner_command = " ".join(
            shlex.quote(item)
            for item in [
                "python3",
                "/opt/data/scripts/run_platform_worker.py",
                *common,
                "--worker-root",
                "/data-aisoft/baoming/hermes/runs/global-workers",
                "--adapter",
                "/opt/data/scripts/run-nvidia-a100-benchmark.sh",
            ]
        )
        launch_log = (
            "/data-aisoft/baoming/infinibench-ops/"
            f"nvidia-launch/{global_run_id}.log"
        )
        outer_command = (
            f"mkdir -p {shlex.quote(str(Path(launch_log).parent))} && "
            + inner_command
            + f" > {shlex.quote(launch_log)} 2>&1"
        )
        command = [
            "docker",
            "exec",
            "-d",
            HERMES,
            "sh",
            "-c",
            outer_command,
        ]
        completed = run_command(command, capture=True)
        if completed.returncode != 0:
            raise RuntimeError(f"NVIDIA launch failed: {completed.stderr.strip()}")
        return {
            "device": device,
            "launch": "docker-exec-detached",
            "returncode": completed.returncode,
        }

    remote_worker = "/data-aisoft/baoming/infinibench-workers/metax-c550/bin/run_platform_worker.py"
    remote_adapter = "/data-aisoft/baoming/infinibench-workers/metax-c550/bin/run-metax-c550-weekly.sh"
    log_dir = "/data-aisoft/baoming/infinibench-ops/remote-launch"
    remote_arguments = [
        "python3",
        remote_worker,
        *common,
        "--worker-root",
        "/data-aisoft/infinibench_data/ops/.workers",
        "--adapter",
        remote_adapter,
    ]
    remote_command = (
        f"mkdir -p {shlex.quote(log_dir)} && "
        + " ".join(shlex.quote(item) for item in remote_arguments)
        + f" > {shlex.quote(f'{log_dir}/{global_run_id}.log')} 2>&1 < /dev/null & "
        + f"echo $! > {shlex.quote(f'{log_dir}/{global_run_id}.pid')}"
    )
    completed = run_command(
        ["ssh", "-o", "BatchMode=yes", METAX_SSH, remote_command]
    )
    completed_output = ((completed.stdout or "") + (completed.stderr or "")).strip()
    if completed.returncode != 0:
        raise RuntimeError(f"MetaX launch failed: {completed_output}")
    return {
        "device": device,
        "launch": "ssh-nohup",
        "returncode": completed.returncode,
        "log": f"{log_dir}/{global_run_id}.log",
        "pid": f"{log_dir}/{global_run_id}.pid",
    }


def command_run_once(args: argparse.Namespace) -> int:
    if is_paused():
        raise SystemExit("paused: remove the global pause with 'infinibenchctl resume'")
    safe_run_id(args.global_run_id)
    if args.dry_run:
        print("dry_run=true")
        print(f"global_run_id={args.global_run_id}")
        print(f"execution_time={args.execution_time}")
        print(f"devices={','.join(args.device)}")
        print(f"dtypes={args.dtypes}")
        print(f"profile={args.profile}")
        print(f"replace_existing_output={args.replace_existing_output}")
        return 0

    create_manifest(args)
    launch_results = []
    errors = []
    for device in args.device:
        try:
            launch_results.append(
                launch_device(
                    device,
                    args.global_run_id,
                    replace_existing_output=args.replace_existing_output,
                )
            )
        except Exception as exc:
            errors.append({"device": device, "error": str(exc)})
    payload = {
        "global_run_id": args.global_run_id,
        "execution_time": args.execution_time,
        "launched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "launches": launch_results,
        "errors": errors,
    }
    atomic_write_json(LAUNCH_ROOT / f"{args.global_run_id}.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if errors else 0


def command_pause(_: argparse.Namespace) -> int:
    payload = {
        "paused": True,
        "reason": "operator requested pause",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    atomic_write_json(pause_marker(), payload)
    print(f"paused=true marker={pause_marker()}")
    print("active workers are not killed; use 'infinibenchctl stop --all' to stop them")
    return 0


def command_resume(_: argparse.Namespace) -> int:
    marker = pause_marker()
    if marker.exists():
        marker.unlink()
    print(f"paused=false removed={marker}")
    return 0


def target_benchmark_processes(container: str, output_dir: str, *, remote: bool = False) -> None:
    # Keep the exact process filter simple and auditable.
    script = r"""
import os, sys
target = sys.argv[1]
me = os.getpid()
for entry in os.scandir('/proc'):
    if not entry.name.isdigit():
        continue
    pid = int(entry.name)
    if pid == me:
        continue
    try:
        cmd = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace')
    except OSError:
        continue
    if 'full_operator_benchmark.py' in cmd and target in cmd:
        try:
            os.kill(pid, 15)
            print(f'terminated pid={pid}')
        except ProcessLookupError:
            pass
"""
    command = ["docker", "exec", container, "python3", "-c", script, "--", output_dir]
    if remote:
        remote_command = " ".join(shlex.quote(item) for item in command)
        command = ["ssh", "-o", "BatchMode=yes", METAX_SSH, remote_command]
    completed = run_command(command)
    output = completed.stdout or ""
    if output.strip():
        print(output.strip())


def mark_worker_stopped(path: Path, reason: str) -> None:
    payload = read_json(path)
    payload.update(
        {
            "state": "stopped",
            "stopped_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "stop_reason": reason,
            "pid": None,
        }
    )
    atomic_write_json(path, payload)


def stop_worker(device: str, global_run_id: str) -> dict[str, Any]:
    path = worker_manifest_path(device, global_run_id)
    if not path.is_file():
        return {"device": device, "global_run_id": global_run_id, "state": "not_found"}
    payload = read_json(path)
    state = payload.get("state")
    result = {
        "device": device,
        "global_run_id": global_run_id,
        "previous_state": state,
        "pid": payload.get("pid"),
    }
    if state != "running":
        result["action"] = "not_running"
        return result

    output_dir = str(payload["benchmark_output"])
    if device == "nvidia-a100":
        target_benchmark_processes(NVIDIA_CONTAINER, output_dir)
        execution_time = str(payload.get("execution_time", ""))
        if execution_time:
            helper = f"baoming_hermes_a100_benchmark_{execution_time}"
            run_command(["docker", "rm", "-f", helper])
        pid = payload.get("pid")
        if pid:
            run_command(["docker", "exec", HERMES, "kill", "-TERM", str(pid)])
    else:
        target_benchmark_processes(METAX_CONTAINER, output_dir, remote=True)
        pid = payload.get("pid")
        if pid:
            run_command(["ssh", "-o", "BatchMode=yes", METAX_SSH, "kill", "-TERM", str(pid)])

    mark_worker_stopped(path, "infinibenchctl stop")
    result["action"] = "stopped"
    result["new_state"] = "stopped"
    return result


def active_worker_runs() -> list[tuple[str, str]]:
    result = []
    for device in DEVICES:
        if not device_worker_root(device).is_dir():
            continue
        for directory in (device_worker_root(device) / device).iterdir():
            path = directory / "worker-manifest.json"
            if not path.is_file():
                continue
            try:
                payload = read_json(path)
            except Exception:
                continue
            if payload.get("state") == "running":
                result.append((device, directory.name))
    return sorted(result)


def command_stop(args: argparse.Namespace) -> int:
    if args.all or args.run_id:
        command_pause(args)
    selected = active_worker_runs()
    if args.run_id:
        selected = [item for item in selected if item[1] == args.run_id]
    elif not args.all:
        raise SystemExit("specify --all or --run-id")
    results = [stop_worker(device, run_id) for device, run_id in selected]
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_cancel_run(args: argparse.Namespace) -> int:
    safe_run_id(args.run_id)
    run_root = RUNS_ROOT / args.run_id
    if not run_root.is_dir():
        raise SystemExit(f"global run not found: {run_root}")
    control = {
        "command": "cancel",
        "requested_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "requested_by": os.environ.get("USER", "unknown"),
    }
    atomic_write_json(run_root / "control.json", control)
    active = active_worker_runs()
    active = [item for item in active if item[1] == args.run_id]
    if not args.no_stop:
        for device, _ in active:
            stop_worker(device, args.run_id)
    print(f"canceled=true manifest={run_root / 'control.json'}")
    print(f"stopped_workers={len(active) if not args.no_stop else 0}")
    return 0


def command_status(_: argparse.Namespace) -> int:
    payload = {
        "paused": is_paused(),
        "pause_marker": str(pause_marker()) if is_paused() else None,
        "active_workers": [
            {"device": device, "global_run_id": run_id}
            for device, run_id in active_worker_runs()
        ],
        "workers": {},
    }
    for device in DEVICES:
        rows = []
        root = device_worker_root(device)
        if root.is_dir():
            for directory in (root / device).iterdir():
                path = directory / "worker-manifest.json"
                if not path.is_file():
                    continue
                try:
                    item = read_json(path)
                except Exception:
                    continue
                rows.append(
                    {
                        "global_run_id": directory.name,
                        "state": item.get("state"),
                        "exit_code": item.get("exit_code"),
                        "attempts": item.get("attempts"),
                        "updated_at": item.get("updated_at"),
                    }
                )
        payload["workers"][device] = sorted(rows, key=lambda row: str(row["global_run_id"]))
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="infinibenchctl")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-once", help="create one run and launch selected workers")
    run.add_argument("--device", choices=DEVICES, action="append", required=True)
    run.add_argument("--global-run-id")
    run.add_argument("--execution-time")
    run.add_argument("--schedule-mode", default="manual")
    run.add_argument("--dtypes", default="float16,float32,bfloat16")
    run.add_argument("--profile", default="decode-4k-small")
    run.add_argument("--infiniops-ref", default="master")
    run.add_argument("--infinirt-ref", default="master")
    run.add_argument("--infiniops-commit")
    run.add_argument("--infinirt-commit")
    run.add_argument("--replace-existing-output", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=command_run_once)

    pause = sub.add_parser("pause", help="prevent new scheduled or manual runs")
    pause.set_defaults(func=command_pause)

    stop = sub.add_parser("stop", help="pause and stop active workers")
    stop.add_argument("--all", action="store_true")
    stop.add_argument("--run-id")
    stop.set_defaults(func=command_stop)

    cancel = sub.add_parser("cancel-run", help="cancel one global run")
    cancel.add_argument("run_id")
    cancel.add_argument("--no-stop", action="store_true", help="only mark canceled; do not stop workers")
    cancel.set_defaults(func=command_cancel_run)

    resume = sub.add_parser("resume", help="remove the global pause marker")
    resume.set_defaults(func=command_resume)

    status = sub.add_parser("status", help="show pause state and worker states")
    status.set_defaults(func=command_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "global_run_id", None) is None:
        now = beijing_now()
        mode = getattr(args, "schedule_mode", "manual")
        args.global_run_id = f"{now:%Y%m%d}-{mode}"
    if getattr(args, "execution_time", None) is None:
        args.execution_time = f"{beijing_now():%Y%m%d-%H%M%S}"
    result = args.func(args)
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
