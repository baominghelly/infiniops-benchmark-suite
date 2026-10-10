# Hygon BW1000/BW1100 benchmark runner guide

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

Run a BW1100 node-2 smoke workflow. Use its node-specific container name while
keeping the logical `DEVICE_ID`; both nodes consequently share one platform
result directory and runner lock:

```bash
CONTAINER=baoming_infiniops_hygon_dcu2 \
DEVICE_ID=hygon-bw1100 \
INFINIBENCH_NODE=hygon-bw1100-02 \
HYGON_ARCH=gfx938 \
DTYPES=float16 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_hygon_benchmark.sh
```

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

If the reused directory already contains `results.json` or `results.jsonl`,
`REUSE_RUN_DIR=1` still refuses to delete those partial or complete results.
Add `REPLACE_RESULT_DIR=1` only when they are explicitly disposable.

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

Before acquiring the lock or changing an output directory, the default
`SOURCE_PREFLIGHT=1` checks that each repository has at least one reachable
URL. Git clone/fetch commands run with inherited proxy variables removed by
`CLEAR_BUILD_PROXY=1`. If a node genuinely needs a proxy, set
`CLEAR_BUILD_PROXY=0`, or provide only trusted values through
`BUILD_HTTP_PROXY`, `BUILD_HTTPS_PROXY`, `BUILD_ALL_PROXY`, and
`BUILD_NO_PROXY`; the runner passes the selected values to Git and pip instead
of propagating an unvalidated container-wide proxy.

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

The runner pins `HYGON_CUDA_ROOT=/opt/dtk/cuda/cuda-12` by default and checks
that its `bin/nvcc`, headers, and `lib64` exist. The current BW1100 image
supplies DTK 26.04 with a CUDA-12-compatible toolkit; BW1100 also pins
`HYGON_ARCH=gfx938`. The runner creates a deterministic staging root such as
`/workspace/.cache/dtk-cuda-12`, links normal DTK components from `/opt/dtk`,
and links `cuda` explicitly to the configured CUDA root. Both projects receive
the same `DTK_ROOT`, `CUDA_HOME`, `CUDA_PATH`, `CMAKE_CUDA_COMPILER`, and
`CUDAToolkit_ROOT`.

Persistent CMake caches are checked before reuse and after configuration. The
runner compares the cached CUDA compiler, toolkit root, Hygon architecture,
and InfiniRT root/install prefix, then refuses a mixed InfiniRT/InfiniOps
toolchain. By default `RESET_STALE_BUILD_DIR=1` removes only an incompatible
cache under `/workspace/.cache/` and reconfigures it. Set
`RESET_STALE_BUILD_DIR=0` when an operator must inspect or repair a cache.

The dependency preflight checks Python build modules and
`python3 -m pybind11 --cmakedir` before compilation. Missing dependencies are
installed with `PYBIND11_VERSION=3.1.0` by default, avoiding a late CMake
failure when `pybind11Config.cmake` is absent.

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

`hygon_gpu_audit.py` obtains data from a host Hygon SMI installation. It tries
`/opt/dtk/bin/rocm-smi`, `/opt/hyhal/bin/hy-smi`, then
`/usr/local/hyhal/bin/hy-smi`; set `ROCM_SMI` to override the executable path.

- process list and HCU ownership from `SMI --showpids`
- utilization from `SMI --showuse`
- VRAM total/used from `SMI --showmeminfo vram`, with
  `SMI --showmemuse` as a percentage-format fallback

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
| `REPLACE_RESULT_DIR` | `0` | Explicitly allow replacing a reused directory that already contains result checkpoints. |
| `INFINIOPS_REF` | Empty | Explicit InfiniOps commit, tag, or branch. |
| `INFINIRT_REF` | Empty | Explicit InfiniRT commit, tag, or branch. |
| `INFINIOPS_REPO_URLS` | GitHub, gh-proxy, ghproxy | Full InfiniOps URL fallback chain. |
| `INFINIRT_REPO_URLS` | GitHub, gh-proxy, ghproxy | Full InfiniRT URL fallback chain. |
| `INFINIOPS_BRANCH` | `master` | InfiniOps remote branch. |
| `INFINIRT_BRANCH` | `master` | InfiniRT remote branch. |
| `UPDATE_SOURCES` | `1` | Prepare both source repositories. |
| `BUILD_INFINIOPS` | `1` | Build InfiniRT and InfiniOps. |
| `HYGON_ARCH` | `gfx906` | Hygon target architecture; use `gfx938` on BW1100. |
| `HYGON_CUDA_ROOT` | `/opt/dtk/cuda/cuda-12` | Exact DTK CUDA-compatible toolkit used by both projects. |
| `HYGON_DTK_ROOT` | `/workspace/.cache/dtk-cuda-12` | Deterministic DTK staging root that forces CUDA discovery. |
| `RESET_STALE_BUILD_DIR` | `1` | Reset an incompatible persistent CMake cache under `/workspace/.cache/`. |
| `CLEAR_BUILD_PROXY` | `1` | Remove inherited proxy variables from Git and pip build commands. |
| `BUILD_HTTP_PROXY` / `BUILD_HTTPS_PROXY` / `BUILD_ALL_PROXY` / `BUILD_NO_PROXY` | Empty | Explicit trusted proxy values for source and dependency downloads. |
| `SOURCE_PREFLIGHT` | `1` | Require one reachable URL per source repository before changing outputs. |
| `PYBIND11_VERSION` | `3.1.0` | Binding build dependency version used when missing or mismatched. |
| `AUTO_SELECT_GPU` | `1` | Automatically select an HCU when no explicit index is supplied. |
| `PHYSICAL_GPU` | Empty | Explicit HCU index; a positional argument takes precedence. |
| `GPU_MAX_UTILIZATION_PERCENT` | `5` | Maximum allowed HCU utilization. |
| `GPU_MIN_FREE_MEMORY_PERCENT` | `0` | Minimum required free VRAM percentage. |
| `GPU_ALLOW_OCCUPIED` | `0` | Permit an already occupied HCU for smoke validation. |
| `ROCM_SMI` | Auto-detected | Host Hygon SMI executable used by the resource gate. |
| `DTYPES` | `float16,float32,bfloat16` | Dtype list. |
| `PROFILES` | `decode-4k` | Profile list. |
| `CONTAINER` | `baoming_infiniops_hygon` | Hygon benchmark container; use a node-unique name on BW1100. |
| `DEVICE_ID` | `hygon-bw1000` | Device directory name. Use the logical `hygon-bw1100` on both BW1100 nodes. |
| `OUTPUT_CSV_NAME` | `infiniops-$DEVICE_ID.csv` | Frontend-facing CSV filename. |
| `OUTPUT_ROOT` | `/data-aisoft/infinibench_data/ops` | Host output root. |
| `CONTAINER_OUTPUT_ROOT` | Same as `OUTPUT_ROOT` | Output root visible in the container. |
| `CONTAINER_BENCHMARK_ROOT` | `/data-aisoft/infinibench-scripts` | Shared tools root visible in the container. |
| `TESTER` | `baoming` | CSV `tester` field. |

## Output contract

For output directory `$RUN_DIR`, important files are:

```text
$RUN_DIR/infiniops-$DEVICE_ID.csv
$RUN_DIR/diagnostics.csv
$RUN_DIR/report.md
$RUN_DIR/benchmark-launcher.log
$RUN_DIR/benchmark-monitor.log
$RUN_DIR/benchmark.done
$RUN_DIR/foreign-process.flag
$RUN_DIR/<dtype>/<profile>/results.json
```

The frontend-facing CSV is derived from `DEVICE_ID`:

```text
infiniops-hygon-bw1000.csv on BW1000
infiniops-hygon-bw1100.csv on both BW1100 nodes
```

The platform directory is deliberately not node-specific. Historical
`hygon-bw1100-01/<date>` data is migrated to `hygon-bw1100/<date>`; node
identity belongs in the worker manifest and benchmark metadata rather than the
result path.

## Daily scheduled run

The daily InfiniBench schedule includes the logical BW1100 platform
`hygon-bw1100`. The A100 control host creates the shared global manifest at
02:00 Beijing time and initially launches node 1 (`hygon251`) over SSH. Both
Hygon nodes disallow user crontabs, so A100 pings both install roots every
five minutes. A fresh shared heartbeat makes the second node exit; a stale
heartbeat or transient failure lets either node acquire the shared logical
worker lock and resume existing checkpoints:

```bash
*/5 * * * * /data-aisoft/baoming/infinibench-ops/periodic_hygon_remote.sh \
  >> /data-aisoft/baoming/infinibench-ops/logs/hygon-periodic.log 2>&1
```

The worker binaries live on shared storage:

```text
/data-aisoft/baoming/infinibench-workers/hygon-bw1100-01/bin/periodic_worker.sh
/data-aisoft/baoming/infinibench-workers/hygon-bw1100-01/bin/periodic_worker.py
/data-aisoft/baoming/infinibench-workers/hygon-bw1100-01/bin/run_platform_worker.py
/data-aisoft/baoming/infinibench-workers/hygon-bw1100-01/bin/run-hygon-bw1100.sh
/data-aisoft/baoming/infinibench-workers/hygon-bw1100-02/bin/periodic_worker.sh
/data-aisoft/baoming/infinibench-workers/hygon-bw1100-02/bin/periodic_worker.py
/data-aisoft/baoming/infinibench-workers/hygon-bw1100-02/bin/run_platform_worker.py
/data-aisoft/baoming/infinibench-workers/hygon-bw1100-02/bin/run-hygon-bw1100.sh
```

The platform environment contract pins the node-specific values:

```text
DEVICE=hygon-bw1100
CONTAINER=baoming_infiniops_hygon on node 1
CONTAINER=baoming_infiniops_hygon_dcu2 on node 2
INFINIBENCH_NODE=hygon-bw1100-01 or hygon-bw1100-02
HYGON_ARCH=gfx938
INFINIOPS_SOURCE_ROOT=/workspace/InfiniOps
INFINIRT_SOURCE_ROOT=/workspace/InfiniRT
```

Daily results are written to:

```text
/data-aisoft/infinibench_data/ops/hygon-bw1100/<YYYYMMDD>/
```

Both BW1100 nodes share this platform output directory and runner lock. They
do not run concurrently: the shared worker lease and heartbeat select one owner,
and takeover is reserved for stale heartbeat or retryable failure cases.

## BW1100 bootstrap notes

The BW1100 nodes expose 8 HCUs with `gfx938:sramecc+:xnack-`. Build with
`HYGON_ARCH=gfx938`. BW1100 installations may expose `hy-smi` under either
`/opt/hyhal/bin` or `/usr/local/hyhal/bin`; `hygon_gpu_audit.py` auto-detects
both locations as well as `/opt/dtk/bin/rocm-smi`.

When the large PyTorch image cannot be fetched completely from Harbor, export
its filesystem from a temporary source-node container and import it on the
target node. The shared helper scripts preserve the image environment,
entrypoint, and working directory:

```bash
docker create --name infinibench-image-export IMAGE
/data-aisoft/infinibench-scripts/scripts/export_hygon_pytorch_image.sh \
  infinibench-image-export |
ssh TARGET_NODE \
  'DECOMPRESS_INPUT=1 /data-aisoft/infinibench-scripts/scripts/import_hygon_pytorch_image.sh IMAGE'
docker rm infinibench-image-export
```

For a fresh imported container, install `nlohmann_json` from the configured
proxy before the first build. If the default PyPI endpoint stalls, install the
build dependencies with an accessible index such as
`https://pypi.tuna.tsinghua.edu.cn/simple`; the persistent container retains
these packages for later runs.

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
selected_source_urls
proxy_sanitized
pybind11_version
hygon_cuda_root
build_cache_reset
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

If all repository candidates are unreachable, the source preflight exits before
the run directory is reset. Check the URL chain or add a reachable local Git
mirror with `INFINIOPS_REPO_URLS` and `INFINIRT_REPO_URLS`.

If dependency installation stalls, inspect the container proxy configuration.
The default build commands intentionally discard stale proxy values. Use the
explicit `BUILD_*_PROXY` variables rather than restoring a machine-wide stale
proxy.

If CMake reports `cudaGraphInstantiate` signature mismatches, unsupported
`gfx938`, or a missing CUDA mocker header, inspect `hygon_cuda_root` and
`build_cache_reset` in `benchmark.done`. A non-empty `build_cache_reset` means
the runner found and removed a stale mixed-toolkit cache. Do not build one
project with CUDA 11 and the other with CUDA 12.
