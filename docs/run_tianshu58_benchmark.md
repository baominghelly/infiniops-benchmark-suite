# Tianshu/Iluvatar benchmark runner guide

This guide describes the host-side orchestrator for the `tianshu58` node with
Iluvatar TG150 accelerators. The canonical device ID is `iluvatar-tg150`.

## Quick start

The formal deployment path is:

```text
/data-aisoft/infinibench-scripts/scripts/run_iluvatar_benchmark.sh
```

Run all default FP16 profiles and automatically select an idle GPU:

```bash
/data-aisoft/infinibench-scripts/scripts/run_iluvatar_benchmark.sh
```

Select a physical GPU explicitly:

```bash
/data-aisoft/infinibench-scripts/scripts/run_iluvatar_benchmark.sh 9
```

Backfill the source versions and CSV date for a historical day:

```bash
EXECUTION_TIME=20260916-120000 \
BACKFILL_DATE=2026-09-08 \
/data-aisoft/infinibench-scripts/scripts/run_iluvatar_benchmark.sh 9
```

The scheduler may also set `PHYSICAL_GPU` instead of passing the GPU index as
the first positional argument.

## Execution model

The runner is a host-side orchestrator. The host shell is responsible only for:

- Docker inspection and `docker exec`
- Iluvatar GPU discovery and resource gating through `ixsmi`
- Machine-level locking
- Output directory creation and path validation
- Scheduling source preparation, compilation, and profiles

All source preparation, compilation, correctness checks, and benchmark
measurement run inside:

```text
infinibench-stack-20260724-iluvatar
```

The container is expected to mount `/data-aisoft` at the same path and to have
CoreX 4.5, the Iluvatar device plugins, Docker access to the GPUs, and Python
3.10 with the CoreX PyTorch installation.

## Time and output naming

`EXECUTION_TIME` defaults to the current local time in `YYYYMMDD-HHMMSS`
format. Its date part is `EXECUTION_DATE`. The complete `EXECUTION_TIME` is
retained as metadata, but the normal output directory uses only the execution
date:

```text
/data-aisoft/infinibench_data/ops/iluvatar-tg150/<EXECUTION_DATE>/
```

For example, `EXECUTION_TIME=20260917-083000` writes to:

```text
/data-aisoft/infinibench_data/ops/iluvatar-tg150/20260917/
```

`BACKFILL_DATE` is an optional historical source date in `YYYY-MM-DD` format.
A backfill uses one target date plus one execution date in its directory name;
the directory does not contain hours, minutes, or seconds:

```text
/data-aisoft/infinibench_data/ops/iluvatar-tg150/backfill-<TARGET_YYYYMMDD>-<EXECUTION_YYYYMMDD>/
```

For example, `BACKFILL_DATE=2026-09-04` and
`EXECUTION_TIME=20260917-083000` write to:

```text
/data-aisoft/infinibench_data/ops/iluvatar-tg150/backfill-20260904-20260917/
```

CSV dates are derived automatically:

- Normal run: the execution date
- Backfill run: `BACKFILL_DATE`

`RUN_DATE` and `CSV_DATE` are rejected. Set `EXECUTION_TIME` when a scheduler
needs to record the exact execution time; the output directory remains
date-only.

`REUSE_RUN_DIR` defaults to `0`. Its only supported values are `0` and `1`.

The date-only directory is deterministic. A non-empty normal execution-date
directory, or a non-empty directory for the same backfill/execution date pair,
causes the runner to exit with code 2 instead of mixing stale checkpoints into
a new attempt. To replace a failed or superseded attempt in the same fixed
directory, set `REUSE_RUN_DIR=1`; the runner deletes and recreates only that
run directory after it holds the machine-level lock.

## Source preparation

The default Git source roots inside the container are:

```text
/workspace/src/infinibench-git/InfiniOps
/workspace/src/infinibench-git/InfiniRT
```

If a repository does not exist, the runner clones it. If it exists, the runner
requires it to be a clean Git repository, fetches the configured branch, and
updates it only with a fast-forward merge. Local changes cause an immediate
failure; the runner never discards or overwrites them.

The default URL order is:

```text
https://github.com/InfiniTensor/<repository>.git
https://gh-proxy.com/https://github.com/InfiniTensor/<repository>.git
https://ghproxy.net/https://github.com/InfiniTensor/<repository>.git
```

Override the complete URL list with `INFINIOPS_REPO_URLS` or
`INFINIRT_REPO_URLS`. Probe and operation timeouts are controlled by
`SOURCE_PROBE_TIMEOUT_SECONDS` and `SOURCE_OPERATION_TIMEOUT_SECONDS`.

For a backfill, each repository selects the last commit on its default branch
before `23:59:59+08:00` on `BACKFILL_DATE`. Explicit `INFINIOPS_REF` and
`INFINIRT_REF` values take precedence over date selection.

## Compilation

Compilation runs entirely through `docker exec`. The runner:

1. Builds InfiniRT for `ivcore20`
2. Exports the selected InfiniOps commit to an isolated build workspace
3. Applies the CoreX 4.5 compatibility patches from
   `/data-aisoft/infinibench-scripts/patches/iluvatar-tg150`
4. Runs the InfiniOps `scripts/dev/build.sh iluvatar` build
5. Uses the isolated install tree for `PYTHONPATH`, `LD_LIBRARY_PATH`, and
   `torch_ops_metadata.json`

The source Git repositories remain clean; generated and patched build sources
are confined to `/workspace/.cache/infinibench-iluvatar` by default. Set
`INFINIBENCH_BUILD_WORKSPACE` to use another absolute container path.

Compilation can be skipped with `BUILD_INFINIOPS=0`, but the caller must then
provide a valid existing install with `INFINIOPS_INSTALL_ROOT` and
`INFINIOPS_METADATA`.

## GPU resource gate

GPU selection uses `ixsmi` and supports these parameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `GPU_MAX_UTILIZATION_PERCENT` | `5` | Maximum compute utilization. |
| `GPU_MIN_FREE_MEMORY_PERCENT` | `0` | Minimum free VRAM percentage. |
| `GPU_ALLOW_OCCUPIED` | `0` | Allow an already occupied GPU. |

The default mode preserves exclusive benchmark semantics: no compute process,
utilization no greater than 5 percent, and at least the configured free-memory
fraction. Automatic selection is refreshed after compilation as well as before
the benchmark preflight. An explicitly requested GPU is never changed.

The final preflight takes three samples. Each sample checks utilization, free
memory percentage, and process count.

### Occupied-GPU mode

Set all three parameters when a workflow validation must run on a GPU that
already has an external process:

```bash
GPU_MAX_UTILIZATION_PERCENT=50 \
GPU_MIN_FREE_MEMORY_PERCENT=10 \
GPU_ALLOW_OCCUPIED=1 \
/data-aisoft/infinibench-scripts/scripts/run_iluvatar_benchmark.sh 9
```

In this mode foreign processes are logged as `foreign_allowed`, the runner does
not write `foreign-process.flag`, and it does not terminate the benchmark. This
mode is only for validating end-to-end workflow behavior. Latency and throughput
from `GPU_ALLOW_OCCUPIED=1` are not comparable with exclusive runs.

## Profiles and dtypes

The default profiles are:

```text
decode-4k,decode-8k,prefill-4k
```

The default dtype is `float16`. Override them with comma-separated lists, for
example:

```bash
DTYPES=float16 PROFILES=decode-4k,decode-8k,prefill-4k \
/data-aisoft/infinibench-scripts/scripts/run_iluvatar_benchmark.sh
```

## Output files

All outputs are written directly to the run directory:

```text
benchmark-launcher.log
benchmark-monitor.log
benchmark.done
foreign-process.flag
diagnostics.csv
report.md
infiniops-iluvatar-biv150.csv
<dtype>/<profile>/results.csv
<dtype>/<profile>/results.json
<dtype>/<profile>/results.jsonl
<dtype>/<profile>/summary.md
```

The final CSV filename is fixed. Successful, correctness-validated cases are
written to that CSV. Failed, unsupported, and reference-rejected cases are
written to `diagnostics.csv`.

`benchmark.done` is non-empty and records:

- `exit_code`
- `foreign_process_seen`
- `execution_time`, `run_date`, `csv_date`, and `backfill_date`
- Requested refs and resolved InfiniOps/InfiniRT commits
- Container, GPU index, GPU UUID, and selection mode
- All three GPU resource-gate parameters and measured free-memory percentage
- Output directory

The runner also has an exit trap that writes a non-empty `benchmark.done` if
source preparation, compilation, or preflight fails after the run directory is
created.

## Locking and NFS permissions

Only one Iluvatar runner may switch source state, compile, or benchmark on the
node at a time. The machine lock is:

```text
/data-aisoft/infinibench_data/ops/iluvatar-tg150/.infinibench-runner.lock
```

The output run directory is created by the container and made sticky. The
host/container path marker is also created and removed by the container, which
avoids cross-account `EPERM` failures on the shared NFS mount.

## Troubleshooting

If source preparation reports local changes, inspect the listed repository with
`git status`; the runner intentionally refuses to overwrite them. If no GPU
passes the gate, lower `GPU_MIN_FREE_MEMORY_PERCENT` or raise
`GPU_MAX_UTILIZATION_PERCENT` only when the semantics are appropriate. Use
`GPU_ALLOW_OCCUPIED=1` only for non-latency workflow validation.
