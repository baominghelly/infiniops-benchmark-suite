# Ascend 910C benchmark runner guide

`scripts/run_ascend_benchmark.sh` is the host-side entry point for InfiniOps and
InfiniRT operator benchmarks on the Ascend 910B machine. The deployed shared
entry point is:

```text
/data-aisoft/infinibench-scripts/scripts/run_ascend_benchmark.sh
```

Run it on the Ascend host, not inside the Docker container. The host shell only
performs Docker, `npu-smi`, filesystem, lock, and scheduling work. All source
preparation, compilation, benchmark execution, and report export run inside the
configured container through `docker exec`.

## Quick start

Run the default exclusive workflow with `float16` and the full `decode-4k`
operator set:

```bash
/data-aisoft/infinibench-scripts/scripts/run_ascend_benchmark.sh
```

Run the reduced development profile:

```bash
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_ascend_benchmark.sh
```

Backfill the source versions that were current on a historical date:

```bash
BACKFILL_DATE=2026-09-11 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_ascend_benchmark.sh
```

Retry immediately after a successful build without rebuilding:

```bash
BUILD_INFINIOPS=0 \
BACKFILL_DATE=2026-09-11 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_ascend_benchmark.sh
```

Only use `BUILD_INFINIOPS=0` when the installed `infini.ops` package was built
from the requested revisions and those revisions have not changed. The runner
does not compare installed wheel contents with the checkout when compilation is
skipped.

## Execution model

One invocation performs these steps:

1. Validate parameters and acquire the machine-level runner lock.
2. Create the canonical output directory on the shared `/data-aisoft` mount.
3. Prepare `/workspace/InfiniOps` and `/workspace/InfiniRT` inside the container.
4. Select a physical NPU with the configured resource gate.
5. Build InfiniOps with Ascend/CANN and Torch support inside the container.
6. Refresh automatic NPU selection after compilation.
7. Run NPU utilization, free-HBM, process, and Python runtime preflight checks.
8. Benchmark every requested dtype/profile combination.
9. Export the fixed-name CSV, diagnostics CSV, and Markdown report.
10. Write non-empty completion metadata to `benchmark.done`.

The machine-level lock prevents two runners from switching the same source
trees or persistent build cache at the same time. A second invocation exits
with code 2.

## Time and output naming

`EXECUTION_TIME` is the actual execution time. It defaults to the current time
and may be overridden as `YYYYMMDD-HHMMSS`. `BACKFILL_DATE` is an optional
source date in `YYYY-MM-DD` format.

A normal output directory uses the detected platform directory:

```text
/data-aisoft/infinibench_data/ops/ascend-910c/<EXECUTION_DATE>/
```

A backfill directory is:

```text
/data-aisoft/infinibench_data/ops/ascend-910c/backfill-<YYYYMMDD>-<EXECUTION_DATE>/
```

The CSV `date` column is the execution date for a normal run and the target
date for a backfill. `RUN_DATE` and `CSV_DATE` are rejected.

Both normal and backfill directories are deterministic for their date keys. A
non-empty directory with the same name causes the runner to exit instead of
mixing stale checkpoints into a new attempt. To replace a failed or superseded
attempt in the same fixed directory, set `REUSE_RUN_DIR=1`; this deletes and
recreates only that run directory after the machine lock is held.

All raw results and final artifacts are written directly to the date directory:

```text
benchmark-launcher.log
benchmark-monitor.log
foreign-process.flag
benchmark.done
<dtype>/<profile>/results.json
<dtype>/<profile>/results.jsonl
<dtype>/<profile>/results.csv
<dtype>/<profile>/summary.md
infiniops-ascend-910c.csv
diagnostics.csv
report.md
```

The host and container both use `/data-aisoft/infinibench_data/ops` as the
canonical output root. The runner verifies the bind mount with a marker created
and removed by the container, which avoids host-side `EPERM` in sticky NFS
directories.
`OUTPUT_ROOT` and `CONTAINER_OUTPUT_ROOT` remain available as overrides.

## Source preparation and backfill

The default source locations inside the container are:

```text
/workspace/InfiniOps
/workspace/InfiniRT
```

If either repository is absent, it is cloned with full history. If it exists,
the runner requires a clean work tree, updates `origin`, fetches branches and
tags, and fast-forwards the configured branch. Dirty trees fail immediately;
local changes are never discarded.

The default URL fallback chain is:

```text
https://github.com/InfiniTensor/<repository>.git
https://gh-proxy.com/https://github.com/InfiniTensor/<repository>.git
https://ghproxy.net/https://github.com/InfiniTensor/<repository>.git
```

`INFINIOPS_REPO_URLS` and `INFINIRT_REPO_URLS` accept comma-separated chains.
`INFINIOPS_REPO_URL` and `INFINIRT_REPO_URL` select one URL explicitly. Probes
use `SOURCE_PROBE_TIMEOUT_SECONDS=30`; clone/fetch operations use
`SOURCE_OPERATION_TIMEOUT_SECONDS=900`.

For a date backfill, each repository checks out the last commit on its
configured remote branch no later than 23:59:59 Asia/Shanghai on the target
date. `INFINIOPS_REF` and `INFINIRT_REF` take precedence over the date. Set both
refs when an exactly matched pair is required.

## NPU resource gate

Automatic selection defaults to exclusive benchmark semantics:

```text
no running NPU process
AICore utilization <= 5%
free HBM >= 90%
```

The equivalent parameters are:

```text
GPU_MAX_UTILIZATION_PERCENT=5
GPU_MIN_FREE_MEMORY_PERCENT=90
GPU_ALLOW_OCCUPIED=0
```

The names retain `GPU` for cross-platform scheduling compatibility. Values are
integer percentages from 0 through 100. Automatic selection is refreshed after
compilation. An explicit positional NPU index or `PHYSICAL_NPU` is never
changed automatically, but it still passes the utilization and free-HBM checks.

For a functional smoke run that may share a device:

```bash
GPU_MAX_UTILIZATION_PERCENT=40 \
GPU_MIN_FREE_MEMORY_PERCENT=40 \
GPU_ALLOW_OCCUPIED=1 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_ascend_benchmark.sh
```

When occupied devices are allowed, external processes are logged as
`foreign_allowed`; the runner does not write `foreign-process.flag` and does not
terminate the benchmark. Latency from this mode is not comparable with
exclusive benchmarks and must only be used for pipeline validation.

## Build and runtime defaults

The default build config enables CPU, Ascend, Torch, and Ascend custom kernels,
and uses the independent `/workspace/InfiniRT` source tree. The persistent build
directory is `/workspace/.cache/infiniops-build`. `CMAKE_BUILD_PARALLEL_LEVEL`
defaults to 32.

The default hardware profile is `Ascend910_9362`, matching the 910C devices
reported by `npu-smi`; the result platform directory is `ascend-910c`. Override
`SOC_VERSION` only with a value supported by the installed CANN toolkit.

The runner preserves the image CANN `LD_LIBRARY_PATH` and appends the installed
Torch and torch_npu library directories for `infini.ops` imports and benchmark
workers.

On `910c_245`, use the host-specific entry point:

```bash
/data-aisoft/infinibench-scripts/scripts/run_ascend910c_245_benchmark.sh <0-15>
```

That machine reports `Ascend910_9362` devices and exposes 16 logical NPUs
(two chips on each of eight physical cards). The wrapper writes under
`ascend-910c/` and builds with `SOC_VERSION=Ascend910_9362`. Omitting the
index automatically selects the lowest-numbered idle logical NPU; passing
0 through 15 pins an explicit chip.

`DTYPES` defaults to `float16`; `float32` and `bfloat16` are also accepted.
`PROFILES` defaults to `decode-4k`; `decode-4k-small`, `decode-8k`, and
`prefill-4k` are accepted. Use `ASCEND_DECODE_4K_OPS`,
`ASCEND_DECODE_4K_SMALL_OPS`, `ASCEND_DECODE_8K_OPS`, and
`ASCEND_PREFILL_4K_OPS` for operator-list overrides.

## Completion metadata

`benchmark.done` is always non-empty after the run directory has been created,
including failure paths. It records at least:

```text
exit_code
stage
foreign_process_seen
end
container
physical_npu
npu_selection
gpu_max_utilization_percent
gpu_min_free_memory_percent
gpu_allow_occupied
execution_time
execution_date
run_date
csv_date
source_backfill_date
infiniops_requested_ref
infinirt_requested_ref
sha
infinirt_sha
```

The launcher log additionally records the selected and rejected NPU candidates,
measured free-HBM percentage, source URL fallback attempts, selected commits,
and build configuration.
