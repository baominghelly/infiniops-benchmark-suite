# Hygon BW1000 benchmark runner guide

`scripts/run_hygon_benchmark.sh` is the host-side entry point for InfiniOps and
InfiniRT benchmarks on the Hygon BW1000/DCU machine. The deployed shared entry
point is:

```text
/data-aisoft/infinibench-scripts/scripts/run_hygon_benchmark.sh
```

Run it on the Hygon host, not inside Docker. The host shell performs Docker,
DCU resource checks, output-directory setup, and scheduling. All Git source
preparation, compilation, testing, and benchmark execution run inside the
configured container through `docker exec`.

## Quick start

Run the default three-dtype full `decode-4k` workflow:

```bash
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k \
/data-aisoft/infinibench-scripts/scripts/run_hygon_benchmark.sh
```

Run the reduced development profile:

```bash
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_hygon_benchmark.sh
```

Backfill the source revisions that were current on a historical date:

```bash
BACKFILL_DATE=2026-09-11 \
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_hygon_benchmark.sh
```

Retry immediately after a successful build without rebuilding:

```bash
BUILD_INFINIOPS=0 \
BACKFILL_DATE=2026-09-11 \
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_hygon_benchmark.sh
```

Only use `BUILD_INFINIOPS=0` when the installed `infini.ops` package was just
built from the requested source revisions and those revisions have not changed.
The runner does not compare installed wheel contents with the source tree when
compilation is skipped.

## Execution model

One invocation performs these steps:

1. Validate arguments and acquire the machine-level Hygon runner lock.
2. Create the canonical output directory under `/data-aisoft/infinibench_data/ops`.
3. Prepare `InfiniOps` and `InfiniRT` inside the container.
4. Select a physical HCU with the configured resource gate.
5. Build and install InfiniRT, then build InfiniOps, inside the container.
6. Refresh automatic HCU selection after compilation.
7. Run GPU and Python/runtime preflight checks.
8. Benchmark every requested dtype/profile combination.
9. Export a combined CSV, diagnostics CSV, and Markdown report.
10. Write non-empty completion metadata to `benchmark.done`.

The lock is:

```text
/data-aisoft/infinibench_data/ops/hygon-bw1000/.infinibench-runner.lock
```

It prevents two runners from simultaneously changing the shared source trees or
persistent build directories. A second invocation exits with code 2.

## Time and output naming

- `EXECUTION_TIME` is the complete actual execution time. It defaults to the
  current time and may be overridden as `YYYYMMDD-HHMMSS`; it is retained in
  metadata.
- `execution_date` is the `YYYYMMDD` date portion of `EXECUTION_TIME`.
- `BACKFILL_DATE` is the optional historical source date in `YYYY-MM-DD`
  format.
- `RUN_DATE` and `CSV_DATE` are rejected.

A normal output directory is:

```text
/data-aisoft/infinibench_data/ops/hygon-bw1000/<EXECUTION_DATE>/
```

A backfill directory is:

```text
/data-aisoft/infinibench_data/ops/hygon-bw1000/backfill-<TARGET-YYYYMMDD>-<EXECUTION-YYYYMMDD>/
```

The CSV `date` field is the execution date for a normal run and the backfill
target date for a backfill run.

A run directory is deterministic: a normal directory represents one execution
date, and a backfill directory represents one target-date/execution-date pair.
Directory names do not include the execution time. A non-empty directory with
the same name causes the runner to exit with code 2 instead of mixing stale
checkpoints into a new attempt. To replace a failed or superseded attempt in
the same fixed directory, set `REUSE_RUN_DIR=1`; after acquiring the machine
lock, the runner deletes and recreates only that run directory. Historical
directories are never moved or deleted.

## Source preparation

The default container source paths are:

```text
/workspace/InfiniOps
/workspace/InfiniRT
```

If a repository is absent, the runner clones it. If it exists, the work tree
must be clean. The runner fetches branches and tags and fast-forwards the
configured branch. Dirty trees fail immediately; local changes are never
discarded.

The default URL fallback chain is:

```text
https://github.com/InfiniTensor/<repository>.git
https://gh-proxy.com/https://github.com/InfiniTensor/<repository>.git
https://ghproxy.net/https://github.com/InfiniTensor/<repository>.git
```

Override the complete chain with `INFINIOPS_REPO_URLS` or
`INFINIRT_REPO_URLS`. `INFINIOPS_REPO_URL` and `INFINIRT_REPO_URL` select one
URL explicitly. Probes use `SOURCE_PROBE_TIMEOUT_SECONDS=30`; clone and fetch
operations use `SOURCE_OPERATION_TIMEOUT_SECONDS=900`.

For date backfill, each repository selects the last commit on its configured
remote branch no later than 23:59:59 Asia/Shanghai on `BACKFILL_DATE`.
Explicit `INFINIOPS_REF` and `INFINIRT_REF` values take precedence over the
date. Set both refs when a matched pair is required.

## Hygon build

The Hygon path uses two build stages:

1. Build and install InfiniRT with `WITH_CPU=ON` and `WITH_HYGON=ON`.
2. Build InfiniOps against that InfiniRT installation with `WITH_CPU=ON`,
   `WITH_HYGON=ON`, `WITH_TORCH=ON`, and `WITH_LINKED=OFF`.

Default paths are:

```text
InfiniRT build:     /workspace/.cache/infinirt-hygon-build
InfiniRT install:   /workspace/.cache/infinirt-hygon-install
InfiniOps build:    /workspace/.cache/infiniops-hygon-build
```

The default architecture is `HYGON_ARCH=gfx906`. The container image supplies
DTK, the Hygon CUDA-compatible compiler, and the required runtime libraries.

## DCU resource gate

The default gate preserves exclusive benchmark semantics:

```text
GPU_MAX_UTILIZATION_PERCENT=5
GPU_MIN_FREE_MEMORY_PERCENT=0
GPU_ALLOW_OCCUPIED=0
```

Automatic selection runs before compilation and is refreshed after compilation.
An explicitly requested HCU is never changed automatically. Post-build preflight
checks the selected HCU again.

`hygon_gpu_audit.py` obtains data from the host `rocm-smi` installation:

- process list and HCU ownership from `rocm-smi --showpids`
- utilization from `rocm-smi --showuse`
- VRAM total/used from `rocm-smi --showmeminfo vram`, with
  `rocm-smi --showmemuse` as a percentage-format fallback

During the benchmark, process cgroups are classified as `benchmark_container`,
`foreign`, or `foreign_allowed`. In exclusive mode a foreign process writes
`foreign-process.flag` and stops the benchmark. With
`GPU_ALLOW_OCCUPIED=1`, an existing process may remain, monitoring records it
as `foreign_allowed`, no foreign flag is written, and the benchmark is not
terminated.

`GPU_ALLOW_OCCUPIED=1` is only for workflow validation. Its latency and
throughput are not comparable with exclusive-mode results and must not be mixed
with normal weekly benchmark data.

## Important variables

| Variable | Default | Meaning |
|---|---|---|
| `EXECUTION_TIME` | Current time | Complete execution timestamp recorded in metadata. |
| `BACKFILL_DATE` | Empty | Historical source date. |
| `REUSE_RUN_DIR` | `0` | Replace the deterministic run directory after the lock is held. |
| `INFINIOPS_REF` | Empty | Explicit InfiniOps commit, tag, or branch. |
| `INFINIRT_REF` | Empty | Explicit InfiniRT commit, tag, or branch. |
| `INFINIOPS_REPO_URLS` | GitHub, gh-proxy, ghproxy | Full InfiniOps URL fallback chain. |
| `INFINIRT_REPO_URLS` | GitHub, gh-proxy, ghproxy | Full InfiniRT URL fallback chain. |
| `INFINIOPS_BRANCH` | `master` | InfiniOps remote branch. |
| `INFINIRT_BRANCH` | `master` | InfiniRT remote branch. |
| `UPDATE_SOURCES` | `1` | Prepare both source repositories. |
| `BUILD_INFINIOPS` | `1` | Build InfiniRT and InfiniOps. |
| `AUTO_SELECT_GPU` | `1` | Automatically select an HCU when no explicit index is supplied. |
| `PHYSICAL_GPU` | Empty | Explicit HCU index; a positional argument takes precedence. |
| `GPU_MAX_UTILIZATION_PERCENT` | `5` | Maximum allowed HCU utilization. |
| `GPU_MIN_FREE_MEMORY_PERCENT` | `0` | Minimum required free VRAM percentage. |
| `GPU_ALLOW_OCCUPIED` | `0` | Permit an already occupied HCU for smoke validation. |
| `DTYPES` | `float16,float32,bfloat16` | Dtype list. |
| `PROFILES` | `decode-4k` | Profile list. |
| `CONTAINER` | `baoming_infiniops_hygon` | Hygon benchmark container. |
| `DEVICE_ID` | `hygon-bw1000` | Device directory name. |
| `OUTPUT_ROOT` | `/data-aisoft/infinibench_data/ops` | Host output root. |
| `CONTAINER_OUTPUT_ROOT` | Same as `OUTPUT_ROOT` | Output root visible in the container. |
| `CONTAINER_BENCHMARK_ROOT` | `/data-aisoft/infinibench-scripts` | Shared tools root visible in the container. |
| `TESTER` | `baoming` | CSV `tester` field. |

## Output contract

For output directory `$RUN_DIR`, important files are:

```text
$RUN_DIR/infiniops-hygon-bw1000.csv
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
infiniops-hygon-bw1000.csv
```

The host and container output roots must resolve to the same bind mount. The
path marker is created and removed by the container, avoiding host-side EPERM
in sticky or shared NFS directories.

A completed run has a non-empty `benchmark.done` containing:

```text
exit_code
foreign_process_seen
execution_time
execution_date
run_date
csv_date
source_backfill_date
infiniops_requested_ref
infinirt_requested_ref
sha
infinirt_sha
physical_gpu
gpu_max_utilization_percent
gpu_min_free_memory_percent
gpu_allow_occupied
```

Test completion with:

```bash
test -s "$RUN_DIR/benchmark.done"
```

## Troubleshooting

If no HCU passes the gate, wait for an exclusive HCU for comparable data. For a
smoke run only, adjust the thresholds and set `GPU_ALLOW_OCCUPIED=1`.

If source preparation fails, inspect both work trees:

```bash
docker exec baoming_infiniops_hygon \
  git -c safe.directory=/workspace/InfiniOps -C /workspace/InfiniOps status --short
docker exec baoming_infiniops_hygon \
  git -c safe.directory=/workspace/InfiniRT -C /workspace/InfiniRT status --short
```

Commit or stash local changes before retrying. The runner intentionally does
not perform destructive cleanup.
