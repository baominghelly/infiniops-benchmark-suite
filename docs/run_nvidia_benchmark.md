# NVIDIA A100 benchmark runner guide

`scripts/run_a100_benchmark.sh` is the host-side entry point for InfiniOps and
InfiniRT benchmarks on the NVIDIA A100 machine. The deployed shared entry point
is:

```text
/data-aisoft/infinibench-scripts/scripts/run_a100_benchmark.sh
```

Run it on the A100 host, not inside the Docker container. The host-side runner
uses `docker`, `nvidia-smi`, `/proc`, and a machine-level lock for orchestration.
Source preparation, compilation, benchmark execution, and report export all run
inside the configured container through `docker exec`.

## Quick start

Run the default workflow with three dtypes and the full `decode-4k` profile:

```bash
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k \
/data-aisoft/infinibench-scripts/scripts/run_a100_benchmark.sh
```

Run the reduced development profile:

```bash
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_a100_benchmark.sh
```

Backfill the code versions that were current on a historical date:

```bash
BACKFILL_DATE=2026-09-11 \
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_a100_benchmark.sh
```

Retry immediately after a successful build without rebuilding:

```bash
BUILD_INFINIOPS=0 \
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_a100_benchmark.sh
```

Only use `BUILD_INFINIOPS=0` when the installed `infini.ops` package was just
built from the requested source revisions and those revisions have not changed.
The runner does not verify installed wheel contents when compilation is skipped.

## Execution model

One invocation performs these steps:

1. Validate parameters and acquire the machine-level runner lock.
2. Create the canonical output directory under `/data-aisoft/infinibench_data/ops`.
3. Prepare `InfiniOps` and `InfiniRT` inside the container.
4. Select an A100 according to the configured GPU resource gate.
5. Build InfiniOps inside the container with the selected InfiniRT source tree.
6. Refresh automatic GPU selection after compilation.
7. Run NVIDIA and Python/runtime preflight checks.
8. Benchmark every requested dtype/profile combination.
9. Export one combined CSV, diagnostics CSV, and Markdown report.
10. Write non-empty completion metadata to `benchmark.done`.

The lock prevents two runners from simultaneously switching the shared
`/workspace/InfiniOps` and `/workspace/InfiniRT` trees or competing for the same
persistent build cache. A second invocation exits with code 2.

## Time and output naming

There are two time concepts:

- `EXECUTION_TIME`: actual execution time. It defaults to the current time and
  can be overridden as `YYYYMMDD-HHMMSS`.
- `BACKFILL_DATE`: optional historical source date in `YYYY-MM-DD` format.

A date-only execution date is derived as the first eight characters of
`EXECUTION_TIME`:

```text
execution_date=${EXECUTION_TIME:0:8}
```

A normal output directory is:

```text
/data-aisoft/infinibench_data/ops/nvidia-a100/<EXECUTION_DATE>/
```

A backfill output directory is:

```text
/data-aisoft/infinibench_data/ops/nvidia-a100/backfill-<YYYYMMDD>-<EXECUTION_DATE>/
```

The CSV `date` column is the actual execution date for a normal run and the
backfill target date for a backfill run. `RUN_DATE` and `CSV_DATE` are not
accepted. Set `EXECUTION_TIME` explicitly when a scheduler needs a deterministic
output path.

The full `EXECUTION_TIME` remains in launcher metadata and `benchmark.done`, but
it is no longer part of the output directory name. A non-empty normal
`<EXECUTION_DATE>` directory or a non-empty backfill target/execution-date pair
causes exit code 2 so stale checkpoints are never mixed into a new attempt. To
replace a failed fixed directory for the same date or date pair, explicitly set
`REUSE_RUN_DIR=1`; the runner deletes and recreates that directory only after
acquiring the machine-level lock.

The runner does not migrate historical results from the legacy
`/data-aisoft/infinibench_data/nvidia-a100/` directory.

## Source preparation and backfill

The default source locations inside the container are:

```text
/workspace/InfiniOps
/workspace/InfiniRT
```

The default repositories are:

```text
https://github.com/InfiniTensor/InfiniOps.git
https://github.com/InfiniTensor/InfiniRT.git
```

If a repository does not exist, the runner clones it. If it already exists, the
runner requires a clean Git work tree, updates `origin`, fetches branches and
tags, and fast-forwards the configured local branch. A dirty work tree fails
immediately; local changes are never discarded.

The default URL fallback chain is:

```text
https://github.com/InfiniTensor/<repository>.git
https://gh-proxy.com/https://github.com/InfiniTensor/<repository>.git
https://ghproxy.net/https://github.com/InfiniTensor/<repository>.git
```

`INFINIOPS_REPO_URLS` and `INFINIRT_REPO_URLS` accept comma-separated chains.
`INFINIOPS_REPO_URL` and `INFINIRT_REPO_URL` select one URL explicitly.

For a date backfill, each repository chooses the last commit on its configured
remote branch whose commit time is no later than 23:59:59 Asia/Shanghai on
`BACKFILL_DATE`, then checks out that commit. Explicit `INFINIOPS_REF` and
`INFINIRT_REF` values take precedence over the date. Set both refs when a matched
pair is required.

`UPDATE_SOURCES=0` skips source preparation and requires both repositories to
already exist. It cannot be combined with `BACKFILL_DATE`, `INFINIOPS_REF`, or
`INFINIRT_REF`.

## GPU resource gate

The default gate preserves exclusive benchmark semantics:

- No existing compute process on the selected GPU.
- GPU utilization no greater than 5%.
- The free-memory threshold is enabled but defaults to 0%.

NVIDIA free memory is calculated from `nvidia-smi` as:

```text
(memory.total - memory.used) / memory.total * 100
```

The controls are:

| Variable | Default | Meaning |
|---|---:|---|
| `GPU_MAX_UTILIZATION_PERCENT` | `5` | Maximum allowed utilization. |
| `GPU_MIN_FREE_MEMORY_PERCENT` | `0` | Minimum required free VRAM percentage. |
| `GPU_ALLOW_OCCUPIED` | `0` | Permit an already occupied GPU. |
| `AUTO_SELECT_GPU` | `1` | Automatically select a GPU when no index is supplied. |
| `PHYSICAL_GPU` | empty | Explicit physical GPU index. |

A positional GPU index takes precedence over `PHYSICAL_GPU`. Automatic selection
samples three times, chooses the lowest-index GPU that passes every sample, and
is refreshed after compilation. An explicit GPU is never changed automatically,
but it still passes the configured preflight gate.

For a functional smoke run that may share a device:

```bash
GPU_MAX_UTILIZATION_PERCENT=40 \
GPU_MIN_FREE_MEMORY_PERCENT=40 \
GPU_ALLOW_OCCUPIED=1 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_a100_benchmark.sh
```

In this mode the monitor labels outside processes as `foreign_allowed`, does
not write `foreign-process.flag`, and does not terminate the benchmark. Such a
run is only suitable for pipeline validation; its latency values are not
comparable with exclusive benchmark results.

## Build

The default container is:

```text
baoming_infiniops
```

The default persistent build directory is:

```text
/workspace/.cache/infiniops-build
```

The build uses:

```text
WITH_CPU=ON
WITH_NVIDIA=ON
WITH_TORCH=ON
WITH_LINKED=OFF
CMAKE_CUDA_ARCHITECTURES=80
```

`CUDA_ARCHITECTURES` can override the architecture list for a different NVIDIA
device. `CMAKE_BUILD_PARALLEL_LEVEL` defaults to 16 on the A100 host.

## Output files

For an output directory named `$RUN_DIR`, the main files are:

```text
$RUN_DIR/infiniops-nvidia-a100.csv
$RUN_DIR/diagnostics.csv
$RUN_DIR/report.md
$RUN_DIR/benchmark-launcher.log
$RUN_DIR/benchmark-monitor.log
$RUN_DIR/benchmark.done
$RUN_DIR/foreign-process.flag
$RUN_DIR/<dtype>/<profile>/results.json
```

The frontend-facing CSV is always named:

```text
infiniops-nvidia-a100.csv
```

The host and container output roots must resolve to the same bind mount. The
runner checks this with a marker created and removed by the container. This
avoids host-side unlink permission failures in the sticky output directory.

`benchmark.done` is created empty at startup and populated at the end. Test
completion with:

```bash
test -s "$RUN_DIR/benchmark.done"
```

It records the exit code, foreign-process result, execution time, CSV date,
backfill date, requested refs, InfiniOps and InfiniRT commits, GPU index and
UUID, and all three GPU resource-gate parameters.

## Monitoring

Watch the launcher log:

```bash
tail -f "$RUN_DIR/benchmark-launcher.log"
```

Watch the monitor log:

```bash
tail -f "$RUN_DIR/benchmark-monitor.log"
```

The monitor attributes compute processes through `/proc/<pid>/cgroup`. Processes
in the benchmark container are allowed. Outside processes are fatal in exclusive
mode and are labeled `foreign_allowed` when `GPU_ALLOW_OCCUPIED=1`.

## Troubleshooting

### GitHub connection times out

Use `INFINIOPS_REPO_URLS` and `INFINIRT_REPO_URLS` to provide a comma-separated
mirror chain. Probe and operation timeouts are controlled by
`SOURCE_PROBE_TIMEOUT_SECONDS` and `SOURCE_OPERATION_TIMEOUT_SECONDS`.

### Source preparation refuses to continue

Inspect both work trees:

```bash
docker exec baoming_infiniops \
  git -c safe.directory=/workspace/InfiniOps -C /workspace/InfiniOps status --short
docker exec baoming_infiniops \
  git -c safe.directory=/workspace/InfiniRT -C /workspace/InfiniRT status --short
```

Commit or stash local changes before retrying. The runner intentionally avoids
destructive cleanup.

### No GPU satisfies the resource gate

Wait for an idle A100 for comparable latency data. For a smoke run only, adjust
the three GPU gate variables as shown above. Do not compare latency from an
occupied-device run.

### A second invocation exits immediately

Another A100 runner holds:

```text
/data-aisoft/infinibench_data/ops/nvidia-a100/.infinibench-runner.lock
```

Wait for the active run instead of removing the lock.
