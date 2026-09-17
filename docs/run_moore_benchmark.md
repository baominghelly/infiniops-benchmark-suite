# Moore Threads MTT S5000 benchmark runner guide

`scripts/run_moore_benchmark.sh` is the host-side entry point for InfiniOps and
InfiniRT operator benchmarks on Moore97. The deployed shared entry point is:

```text
/data-aisoft/infinibench-scripts/scripts/run_moore_benchmark.sh
```

Run it on the Moore host, not inside the Docker container. The host shell uses
`docker`, `mthreads-gmi`, `/proc`, and a machine-level lock for orchestration.
All source preparation, compilation, profiling, and benchmark work run inside
the configured container through `docker exec`.

Only two deployment roots are used on the shared host filesystem:

```text
/data-aisoft/infinibench-scripts
/data-aisoft/infinibench_data
```

InfiniOps and InfiniRT live inside the container and are not cloned to the
shared host filesystem.

## Quick start

Run the default workflow with three dtypes and the full `decode-4k` profile:

```bash
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k \
/data-aisoft/infinibench-scripts/scripts/run_moore_benchmark.sh
```

Run the reduced development profile:

```bash
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_moore_benchmark.sh
```

Backfill the source versions that were current on a historical date:

```bash
BACKFILL_DATE=2026-09-11 \
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_moore_benchmark.sh
```

Retry immediately after a successful build without rebuilding:

```bash
BUILD_INFINIOPS=0 \
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_moore_benchmark.sh
```

Only set `BUILD_INFINIOPS=0` when the installed `infini.ops` package was built
from the requested revisions and those revisions have not changed. The runner
does not compare installed wheel contents with the source trees when the build
stage is skipped.

## Execution model

One invocation performs these steps:

1. Validate parameters and acquire the Moore machine-level lock.
2. Create the canonical output directory under `/data-aisoft/infinibench_data`.
3. Prepare `/workspace/InfiniOps` and `/workspace/InfiniRT` inside the container.
4. Select an MTT S5000 according to the configured resource gate.
5. Build InfiniOps inside the container with the selected InfiniRT source.
6. Refresh automatic GPU selection after compilation.
7. Run utilization, free-memory, process, MUSA, Python, and InfiniOps preflight.
8. Benchmark each requested dtype/profile combination.
9. Export one combined CSV, diagnostics CSV, and Markdown report.
10. Write non-empty completion metadata to `benchmark.done`.

The machine-level lock prevents two Moore runners from switching the shared
container source trees or competing for the persistent build cache. If another
run is active, the second invocation exits with code 2.

## Time and output naming

There are two time concepts:

- `EXECUTION_TIME`: actual execution time. It defaults to the current time and
  can be overridden as `YYYYMMDD-HHMMSS`.
- `BACKFILL_DATE`: optional historical source date in `YYYY-MM-DD` format.

A normal output directory uses the execution date without the time:

```text
/data-aisoft/infinibench_data/ops/moore-s5000/<EXECUTION_DATE>/
```

For example:

```text
/data-aisoft/infinibench_data/ops/moore-s5000/20260916/
```

A backfill output directory is deterministic for the target date and execution
date:

```text
/data-aisoft/infinibench_data/ops/moore-s5000/backfill-<YYYYMMDD>-<EXECUTION_DATE>/
```

For example:

```text
/data-aisoft/infinibench_data/ops/moore-s5000/backfill-20260911-20260916/
```

The CSV `date` column is the execution date for a normal run and the backfill
target date for a backfill run. `RUN_DATE` and `CSV_DATE` are rejected. The
complete `EXECUTION_TIME` remains in launcher metadata and `benchmark.done` for
provenance even though it is not part of the output directory name.

A normal run directory is deterministic by execution date, and a backfill
directory is deterministic by its target-date/execution-date pair. A non-empty
directory with the same name causes the runner to exit instead of mixing stale
checkpoints into a new attempt. To replace a failed or superseded attempt in
the same fixed directory, set `REUSE_RUN_DIR=1`; this deletes and recreates only
that run directory after the machine lock is held.

## Source preparation and backfill

The default source locations inside the container are:

```text
/workspace/InfiniOps
/workspace/InfiniRT
```

If a repository is absent, it is cloned with full history. If it already exists,
the runner requires a clean Git work tree, updates `origin`, fetches branches
and tags, and fast-forwards the configured local branch. A dirty work tree
causes an immediate failure. The runner never discards local changes.

Because direct GitHub access can be intermittent, the default URL chain is:

```text
https://github.com/InfiniTensor/<repository>.git
https://gh-proxy.com/https://github.com/InfiniTensor/<repository>.git
https://ghproxy.net/https://github.com/InfiniTensor/<repository>.git
```

`INFINIOPS_REPO_URLS` and `INFINIRT_REPO_URLS` accept comma-separated fallback
chains. `INFINIOPS_REPO_URL` and `INFINIRT_REPO_URL` select one URL explicitly.

For a date backfill, each repository chooses the last commit on its configured
remote branch whose commit time is no later than 23:59:59 Asia/Shanghai on
`BACKFILL_DATE`, then checks out that commit. Explicit `INFINIOPS_REF` and/or
`INFINIRT_REF` values take precedence over the date. Set both refs when an exact
repository pair is required.

The selected URL and commits are recorded in `benchmark-launcher.log` and
`benchmark.done`.

## Compilation

Compilation is enabled by default. It runs entirely inside the Moore container
and uses:

```text
WITH_CPU=ON
WITH_MOORE=ON
WITH_TORCH=ON
WITH_LINKED=OFF
GENERATE_PYTHON_BINDINGS=ON
```

The persistent per-commit build cache is inside the container:

```text
/workspace/.cache/infinibench-moore/infiniops-<commit>/
```

The installed metadata and Python extension are written under:

```text
/workspace/.cache/infinibench-moore/infiniops-<commit>/install/
```

The default parallelism is:

```text
CMAKE_BUILD_PARALLEL_LEVEL=16
```

The latest InfiniOps PagedAttention prefill pipelined kernel can require more
shared memory than the MUSA `mp_21` target permits. The runner keeps
commit-era compatibility patches and selects the first one that applies cleanly,
then records its path and SHA256 in the launcher log and build marker. It reverts
the selected patch after the build so the Git work tree remains clean.

## GPU resource gates

Automatic selection priority is:

```text
positional GPU argument > PHYSICAL_GPU > automatic selection
```

The default exclusive-device gate requires:

- no existing GPU process;
- compute utilization no greater than 5%;
- free-memory ratio no less than the configured minimum.

Automatic selection is performed before compilation and refreshed after
compilation. An explicit GPU request is never changed automatically. Preflight
after compilation applies the same thresholds to the selected physical GPU.

For a functional smoke run that may share a device:

```bash
GPU_MAX_UTILIZATION_PERCENT=50 \
GPU_MIN_FREE_MEMORY_PERCENT=50 \
GPU_ALLOW_OCCUPIED=1 \
BUILD_INFINIOPS=0 \
DTYPES=float16 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_moore_benchmark.sh
```

`GPU_ALLOW_OCCUPIED=1` allows an existing foreign process and records it as
`foreign_allowed`; it does not write `foreign-process.flag` and does not
terminate the benchmark. This mode is only for workflow validation. Its latency
data must not be compared with exclusive benchmark results.

`mthreads-gmi -q --json` reports used and total memory separately. The runner
computes:

```text
free_memory_percent = (total MiB - used MiB) / total MiB * 100
```

Compute utilization comes from `Utilization.Gpu`. `mthreads-gmi
--process-monitor` can transiently report a just-exited PID or `pid=0` during
worker transitions. The monitor filters `pid=0`, records an exited PID as
`exited_before_attribution`, and invalidates a run only after three consecutive
resource-query failures.

## Profiles and dtypes

`DTYPES` accepts a comma-separated subset of:

```text
float16,float32,bfloat16
```

`PROFILES` accepts a comma-separated subset of:

```text
decode-4k
decode-4k-small
decode-8k
prefill-4k
```

`decode-4k` is the normal operator-coverage profile. `decode-4k-small` uses
smaller shapes and a representative operator set for development and workflow
validation. `decode-8k` and `prefill-4k` are optional shape-specific profiles.

## Parameter reference

| Parameter | Default | Meaning |
|---|---:|---|
| `DTYPES` | `float16,float32,bfloat16` | Dtypes benchmarked in one invocation. |
| `PROFILES` | `decode-4k` | Profiles benchmarked in one invocation. |
| `EXECUTION_TIME` | Current local time | Actual execution time metadata, `YYYYMMDD-HHMMSS`. |
| `BACKFILL_DATE` | empty | Historical source date, `YYYY-MM-DD`. |
| `REUSE_RUN_DIR` | `0` | Allow the deterministic run directory to be replaced after the machine lock is held. |
| `BUILD_INFINIOPS` | `1` | Build and install InfiniOps before benchmarking. |
| `UPDATE_SOURCES` | `1` | Clone/fetch/check out source repositories. |
| `INFINIOPS_REF` | empty | Explicit InfiniOps commit, tag, or branch. |
| `INFINIRT_REF` | empty | Explicit InfiniRT commit, tag, or branch. |
| `INFINIOPS_REPO_URLS` | GitHub plus two proxies | Comma-separated InfiniOps URL fallback chain. |
| `INFINIRT_REPO_URLS` | GitHub plus two proxies | Comma-separated InfiniRT URL fallback chain. |
| `INFINIOPS_BRANCH` | `master` | Configured InfiniOps branch. |
| `INFINIRT_BRANCH` | `master` | Configured InfiniRT branch. |
| `SOURCE_PROBE_TIMEOUT_SECONDS` | `30` | Per-URL reachability probe timeout. |
| `SOURCE_OPERATION_TIMEOUT_SECONDS` | `900` | Clone/fetch timeout. |
| `AUTO_SELECT_GPU` | `1` | Automatically select a physical GPU when no explicit index is supplied. |
| `PHYSICAL_GPU` | empty | Explicit physical GPU index. A positional argument takes precedence. |
| `GPU_MAX_UTILIZATION_PERCENT` | `5` | Maximum allowed compute utilization. |
| `GPU_MIN_FREE_MEMORY_PERCENT` | `0` | Minimum required free VRAM percentage. |
| `GPU_ALLOW_OCCUPIED` | `0` | Allow an existing process on the selected GPU. |
| `GPU_IDLE_SAMPLES` | `3` | Samples used for selection and preflight. |
| `GPU_IDLE_INTERVAL_SECONDS` | `2` | Interval between idle samples. |
| `GPU_MONITOR_INTERVAL_SECONDS` | `2` | Interval between process-attribution samples. |
| `CMAKE_BUILD_PARALLEL_LEVEL` | `16` | Ninja parallel build level. |
| `INFINIOPS_BUILD_CACHE_ROOT` | `/workspace/.cache/infinibench-moore` | Container-local per-commit build cache. |
| `CONTAINER` | `baoming_infiniops_moore` | Docker container used for compilation and benchmarks. |
| `DEVICE_ID` | `moore-s5000` | Device directory under the output root. |
| `OUTPUT_ROOT` | `/data-aisoft/infinibench_data/ops` | Host canonical output root for operator benchmarks. |
| `CONTAINER_OUTPUT_ROOT` | Same as `OUTPUT_ROOT` | Output root visible inside the container. |
| `CONTAINER_BENCHMARK_ROOT` | `/data-aisoft/infinibench-scripts` | Shared benchmark tool root inside the container. |
| `INFINIOPS_SOURCE_ROOT` | `/workspace/InfiniOps` | InfiniOps source path inside the container. |
| `INFINIRT_SOURCE_ROOT` | `/workspace/InfiniRT` | InfiniRT source path inside the container. |
| `TESTER` | `baoming` | `tester` value written to the exported CSV. |

Invalid parameters, a dirty source tree, an unavailable container, another
active runner, or no GPU satisfying the resource gate normally produce exit code
2. A foreign process on the selected GPU in exclusive mode produces exit code 3.
Build, benchmark, and export failures use the underlying command's nonzero exit
code.

## Output files

For an output directory named `$RUN_DIR`, the main files are:

```text
$RUN_DIR/infiniops-moore-s5000.csv
$RUN_DIR/diagnostics.csv
$RUN_DIR/report.md
$RUN_DIR/benchmark-launcher.log
$RUN_DIR/benchmark-monitor.log
$RUN_DIR/build-infiniops.log
$RUN_DIR/benchmark.done
$RUN_DIR/foreign-process.flag
$RUN_DIR/<dtype>/<profile>/results.json
```

The frontend-facing CSV is always named:

```text
infiniops-moore-s5000.csv
```

The host and container output roots must resolve to the same bind mount. The
runner checks this with a marker created and removed by the container so shared
accounts do not need permission to unlink files from the sticky output
directory.

`benchmark.done` is created empty at startup and populated at completion. Test
completion with:

```bash
test -s "$RUN_DIR/benchmark.done"
```

It records at least:

```text
exit_code
foreign_process_seen
execution_time
csv_date
backfill_date
infiniops_requested_ref
infinirt_requested_ref
infiniops_commit
infinirt_commit
device_id
accelerator
container
physical_gpu
gpu_uuid
gpu_max_utilization_percent
gpu_min_free_memory_percent
gpu_allow_occupied
gpu_resource_mode
latency_comparable
final_csv
```

## Troubleshooting

### GitHub connection times out

The runner tries GitHub directly, then `gh-proxy.com`, then `ghproxy.net`. Use
`INFINIOPS_REPO_URLS` and `INFINIRT_REPO_URLS` to provide a different complete
fallback chain.

### Source preparation refuses to continue

Inspect both container work trees:

```bash
docker exec baoming_infiniops_moore \
  git -c safe.directory=/workspace/InfiniOps -C /workspace/InfiniOps status --short
docker exec baoming_infiniops_moore \
  git -c safe.directory=/workspace/InfiniRT -C /workspace/InfiniRT status --short
```

Commit, stash, or remove local changes before retrying. The runner never
performs a destructive reset.

### No GPU satisfies the resource gate

Wait for an exclusive GPU for comparable latency data. For a smoke run only,
raise `GPU_MAX_UTILIZATION_PERCENT`, set the required free-memory ratio, and set
`GPU_ALLOW_OCCUPIED=1`. Do not compare latency from that run with exclusive
results.

### A second invocation exits immediately

Another Moore runner holds:

```text
/data-aisoft/infinibench_data/ops/moore-s5000/.infinibench-runner.lock
```

Wait for the active run instead of removing the lock manually.

### Permission denied in the output directory

The runner creates and removes its path-check marker inside the container. If
errors remain, verify that `/data-aisoft` is mounted at the same path in the
container and that the invoking account can execute Docker and `mthreads-gmi`.
