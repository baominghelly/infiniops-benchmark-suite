# NVIDIA A100 Known-Failure Doctor Implementation Plan

> **For Codex:** Implement in this session task-by-task. Do not delegate.

**Goal:** Add a deterministic failure-recovery layer for the A100 Hermes workflow so known failures are handled without an LLM and only unknown or exhausted failures request LLM triage.

**Architecture:** The benchmark wrapper records a machine-readable final outcome. A no-agent Doctor cron classifies the latest failed run, applies a small whitelist of idempotent recovery actions, and writes durable state. A monitor script gates LLM triage on `needs_llm=true`; no model credentials are required for known-failure handling.

**Tech Stack:** Bash for the benchmark wrapper, Python 3 standard library for Doctor and monitor, Hermes cron, unittest with temporary directories.

---

### Task 1: Record Final Failure Metadata

**Files:**
- Modify: `hermes/run-nvidia-a100-benchmark.sh`
- Test: `tests/test_hermes_workflow_contract.py`

Write a failing test asserting that the wrapper emits `failure.json`, records final exit code, final console, execution date, and success state. Add a `write_outcome` helper and call it after exclusive and shared attempts. Successful runs write `success.json`; failed runs write `failure.json`.

### Task 2: Classify Known Failures

**Files:**
- Create: `hermes/nvidia_a100_doctor.py`
- Test: `tests/test_nvidia_a100_doctor.py`

Implement pure classification over `failure.json`, console logs, `benchmark.done`, and health probes. Cover `container_cuda_lost`, `no_idle_gpu`, `foreign_process`, `disk_full`, `source_dirty`, and `output_dir_nonempty`. Unknown failures map to `unknown`.

### Task 3: Apply Whitelisted Recovery Playbooks

**Files:**
- Modify: `hermes/nvidia_a100_doctor.py`
- Test: `tests/test_nvidia_a100_doctor.py`

Implement typed actions only:

- `docker_restart_benchmark_container`: only after host GPU works, container is running, no benchmark process is present, and CUDA remains unavailable; verify CUDA after restart.
- `prune_hermes_run_logs`: only under the Hermes runs root, only files older than retention, only up to a byte cap, never current run or benchmark output.
- `capture_source_dirty_state`: export status and diff without mutating Git.
- `schedule_retry`: record retry environment and cooldown; Doctor does not spawn a benchmark during classification tests.

Never kill foreign processes, reset Git, clean benchmark results, or modify shared platform runners.

### Task 4: Add LLM Triage Gate

**Files:**
- Create: `hermes/nvidia_a100_llm_monitor.py`
- Test: `tests/test_nvidia_a100_llm_monitor.py`

Select the latest Doctor state. Emit stable `NO_LLM_NEEDED` unless `needs_llm` is true. For triage, emit compact JSON containing run root, failure code, action history path, and relevant logs. This script contains no API key.

### Task 5: Deploy and Validate

**Files:**
- Modify: `hermes/README.md`

Run syntax checks and targeted unit tests locally. Copy Doctor and monitor into the Hermes container through container-local `/tmp` to avoid NFS `lchown`. Validate Doctor in `--dry-run` against the existing successful 20260917 run. Create an active no-agent Doctor cron every five minutes. Do not create an active LLM cron until the user configures a model API key.

### Task 6: Document LLM Activation

**Files:**
- Modify: `hermes/README.md`

Document model credential setup, safe triage cron command, state paths, known-failure behavior, and security boundaries. Keep API keys out of Git and chat.
