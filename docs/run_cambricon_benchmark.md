# Cambricon MLU590 benchmark runner guide

`scripts/run_cambricon_benchmark.sh` is the host-side entry point for InfiniOps
and InfiniRT benchmarks on the Cambricon MLU590 machine. The deployed shared
entry point is:

```text
/data-aisoft/infinibench-scripts/scripts/run_cambricon_benchmark.sh
```

Run it on the Cambricon host, not inside the Docker container. The host side
uses `docker`, `cnmon`, `/proc`, and a machine-level lock for orchestration; all
source preparation, compilation, benchmark execution, and report export run
inside the configured container through `docker exec`.

## Quick start

Run the default workflow with three dtypes and the full `decode-4k` profile:

```bash
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k \
/data-aisoft/infinibench-scripts/scripts/run_cambricon_benchmark.sh
```

Run the reduced development profile:

```bash
DTYPES=float16 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_cambricon_benchmark.sh
```

Backfill the code versions that were current on a historical date:

```bash
BACKFILL_DATE=2026-09-11 \
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_cambricon_benchmark.sh
```

Retry immediately after a successful build without rebuilding:

```bash
BUILD_INFINIOPS=0 \
BACKFILL_DATE=2026-09-11 \
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_cambricon_benchmark.sh
```

Only use `BUILD_INFINIOPS=0` when the installed `infini.ops` package was just
built from the requested revisions and those revisions have not changed. The
runner does not compare installed wheel contents with the work tree when the
build stage is skipped.

## Execution model

One invocation performs these steps:

1. Validate parameters and acquire the machine-level runner lock.
2. Create the canonical output directory under `/data-aisoft/infinibench_data`.
3. Prepare `InfiniOps` and `InfiniRT` inside the container.
4. Select a physical MLU according to the configured resource gate.
5. Build InfiniOps inside the container with the selected InfiniRT source.
6. Refresh automatic MLU selection after compilation.
7. Run `cnmon` and Python/runtime preflight checks.
8. Benchmark every requested dtype/profile combination.
9. Export one combined CSV, diagnostics CSV, and Markdown report.
10. Write non-empty completion metadata to `benchmark.done`.

The machine-level lock prevents two runners from simultaneously switching the
shared `/workspace/InfiniOps` and `/workspace/InfiniRT` trees or competing for
the persistent build cache. A second invocation exits with code 2.

## Time and output naming

There are two time concepts:

- `EXECUTION_TIME`: actual execution time. It defaults to the current time and
  can be overridden as `YYYYMMDD-HHMMSS`.
- `BACKFILL_DATE`: optional historical source date in `YYYY-MM-DD` format.

A normal output directory is:

```text
/data-aisoft/infinibench_data/ops/cambricon-mlu590/<EXECUTION_DATE>/
```

For example:

```text
/data-aisoft/infinibench_data/ops/cambricon-mlu590/20260916/
```

A backfill output directory is:

```text
/data-aisoft/infinibench_data/ops/cambricon-mlu590/backfill-<YYYYMMDD>-<EXECUTION_DATE>/
```

For example:

```text
/data-aisoft/infinibench_data/ops/cambricon-mlu590/backfill-20260911-20260916/
```

The CSV `date` column is the execution date for a normal run and the backfill
target date for a backfill run. `RUN_DATE` and `CSV_DATE` are not accepted. If
a scheduler needs a deterministic output path, set `EXECUTION_TIME` explicitly
before launching the command.

The complete `EXECUTION_TIME` remains metadata in the launcher log and
`benchmark.done`, but directory names use only `YYYYMMDD`. A non-empty normal
directory for an execution date, or a non-empty backfill directory for a
target/execution date pair, is not reused silently. To replace a failed or
superseded attempt in the same fixed directory, set `REUSE_RUN_DIR=1`; this
deletes and recreates only that date directory after the runner lock is held.
Historical directories outside the current run directory are never moved or
deleted.

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

If a repository does not exist, it is cloned with full history. If it exists,
the runner requires a clean Git work tree, updates `origin`, fetches branches
and tags, and fast-forwards the configured local branch. A dirty work tree
fails immediately; the runner never discards local changes.

Because direct GitHub access can be intermittent, the fallback chain is:

```text
https://github.com/InfiniTensor/<repository>.git
https://gh-proxy.com/https://github.com/InfiniTensor/<repository>.git
https://ghproxy.net/https://github.com/InfiniTensor/<repository>.git
```

`INFINIOPS_REPO_URLS` and `INFINIRT_REPO_URLS` accept comma-separated chains.
`INFINIOPS_REPO_URL` and `INFINIRT_REPO_URL` select one URL explicitly.

For a date backfill, each repository selects the last commit on its configured
remote branch whose commit time is no later than 23:59:59 Asia/Shanghai on
`BACKFILL_DATE`, then checks out that commit. Explicit `INFINIOPS_REF` and/or
`INFINIRT_REF` values take precedence over the date. Set both refs when a
matched pair is required.

`UPDATE_SOURCES=0` skips source preparation and requires both repositories to
already exist. It cannot be combined with `BACKFILL_DATE`, `INFINIOPS_REF`, or
`INFINIRT_REF`.

## MLU resource gate

Automatic selection priority is:

```text
positional MLU argument > PHYSICAL_MLU > automatic selection
```

The default exclusive-device gate requires:

- no existing process on the MLU;
- compute utilization no greater than 5%;
- free-memory ratio no less than the configured minimum.

`cnmon all` reports the card summary and process table, while `cnmon info -m`
reports physical memory `Total`, `Used`, `Reserved`, and `Free`. The runner
uses the detailed `Free/Total` ratio so reserved memory is not counted as
available. Automatic selection is
performed before compilation and refreshed after compilation. An explicit MLU
request is never changed automatically, but it is checked with the same
resource gate before benchmark preflight.

The three resource controls are:

| Variable | Default | Meaning |
|---|---:|---|
| `GPU_MAX_UTILIZATION_PERCENT` | `5` | Maximum allowed MLU utilization. |
| `GPU_MIN_FREE_MEMORY_PERCENT` | `0` | Minimum required free-memory percentage. |
| `GPU_ALLOW_OCCUPIED` | `0` | Allow an existing process on the selected MLU. |

The names intentionally match the cross-platform MetaX runner. For Cambricon,
they apply to physical MLU cards reported by `cnmon`.

For a functional smoke run that may share a device:

```bash
GPU_MAX_UTILIZATION_PERCENT=50 \
GPU_MIN_FREE_MEMORY_PERCENT=50 \
GPU_ALLOW_OCCUPIED=1 \
BUILD_INFINIOPS=0 \
DTYPES=float16 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_cambricon_benchmark.sh
```

`GPU_ALLOW_OCCUPIED=1` allows an existing foreign process and the monitor
records it as `foreign_allowed`. It does not write `foreign-process.flag` and
does not terminate the benchmark. Such a run is useful for pipeline validation,
but its latency data must not be compared with exclusive benchmark results. The
launcher log and `benchmark.done` record `latency_comparable=0` in this mode;
exclusive runs record `latency_comparable=1`.

## Environment

| Variable | Default | Meaning |
|---|---:|---|
| `DTYPES` | `float16,float32,bfloat16` | Comma-separated dtype list. |
| `PROFILES` | `decode-4k` | Comma-separated profile list. |
| `EXECUTION_TIME` | Current `YYYYMMDD-HHMMSS` | Deterministic execution timestamp. |
| `BACKFILL_DATE` | empty | Historical source date. |
| `INFINIOPS_REF` | empty | Explicit InfiniOps commit, tag, or branch. |
| `INFINIRT_REF` | empty | Explicit InfiniRT commit, tag, or branch. |
| `AUTO_SELECT_MLU` | `1` | Automatically select a physical MLU. |
| `PHYSICAL_MLU` | empty | Explicit physical MLU index. A positional argument wins. |
| `GPU_MAX_UTILIZATION_PERCENT` | `5` | Maximum compute utilization. |
| `GPU_MIN_FREE_MEMORY_PERCENT` | `0` | Minimum free-memory percentage. |
| `GPU_ALLOW_OCCUPIED` | `0` | Allow an already occupied MLU for smoke validation. |
| `REUSE_RUN_DIR` | `0` | Replace an existing non-empty fixed date directory explicitly. |
| `CMAKE_BUILD_PARALLEL_LEVEL` | `32` | Ninja parallel build level. |
| `INFINIOPS_BUILD_DIR` | `/workspace/.cache/infiniops-build` | Persistent build directory. |
| `CONTAINER` | `infinicore-cambricon-main-benchmark` | Compilation and benchmark container. |
| `DEVICE_ID` | `cambricon-mlu590` | Device directory under the output root. |
| `OUTPUT_ROOT` | `/data-aisoft/infinibench_data` | Host output root. |
| `CONTAINER_OUTPUT_ROOT` | Same as `OUTPUT_ROOT` | Container output root. |
| `CONTAINER_BENCHMARK_ROOT` | `/data-aisoft/infinibench-scripts` | Shared tools root. |
| `INFINIOPS_SOURCE_ROOT` | `/workspace/InfiniOps` | InfiniOps source inside the container. |
| `INFINIRT_SOURCE_ROOT` | `/workspace/InfiniRT` | InfiniRT source inside the container. |
| `TESTER` | `baoming` | `tester` value in the exported CSV. |

Invalid parameters, a dirty source tree, an unavailable container, another
active runner, or no MLU satisfying the gate normally produce exit code 2. A
foreign process in exclusive mode produces exit code 3. Build, benchmark, and
export failures use the underlying command's nonzero exit code.

## Output and completion

For an output directory named `$RUN_DIR`, the main files are:

```text
$RUN_DIR/infiniops-cambricon-mlu590.csv
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
infiniops-cambricon-mlu590.csv
```

The host and container output roots must resolve to the same bind mount. The
runner verifies this with a marker created and removed by the container, so a
shared account does not need permission to unlink files in the sticky output
directory.

`benchmark.done` is created empty at startup and populated only at the end.
Test completion with:

```bash
test -s "$RUN_DIR/benchmark.done"
```

It records `exit_code`, `foreign_process_seen`, `execution_time`, `csv_date`,
backfill date, requested refs, InfiniOps and InfiniRT commits, selected MLU,
the three resource-gate values, actual utilization and free-memory percentage,
and whether latency is comparable.

## Troubleshooting

### GitHub connection times out

Use `INFINIOPS_REPO_URLS` and `INFINIRT_REPO_URLS` to provide an internal
comma-separated mirror chain.

### Source preparation refuses to continue

Inspect both work trees:

```bash
docker exec infinicore-cambricon-main-benchmark \
  git -c safe.directory=/workspace/InfiniOps -C /workspace/InfiniOps status --short
docker exec infinicore-cambricon-main-benchmark \
  git -c safe.directory=/workspace/InfiniRT -C /workspace/InfiniRT status --short
```

Commit or stash local changes before retrying. The runner intentionally avoids
destructive cleanup.

### No MLU satisfies the resource gate

Wait for an exclusive MLU for comparable latency data. For a smoke run only,
raise `GPU_MAX_UTILIZATION_PERCENT`, adjust
`GPU_MIN_FREE_MEMORY_PERCENT`, and set `GPU_ALLOW_OCCUPIED=1`. Shared-device
latency is not comparable.

### A second invocation exits immediately

Another Cambricon runner is active and holds:

```text
/data-aisoft/infinibench_data/ops/cambricon-mlu590/.infinibench-runner.lock
```

Wait for the active run instead of removing the lock manually.
