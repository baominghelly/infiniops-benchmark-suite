# MetaX C550 benchmark runner guide

`scripts/run_metax_benchmark.sh` is the host-side entry point for InfiniOps and
InfiniRT benchmarks on the MetaX C550 machine. The deployed shared entry point is:

```text
/data-aisoft/infinibench-scripts/scripts/run_metax_benchmark.sh
```

Run it on the MetaX host, not inside the Docker container. The script uses host
`docker`, `mx-smi`, `/proc`, and a machine-level lock for orchestration; all
source preparation, compilation, and benchmark work run inside the configured
container through `docker exec`.

## Quick start

Run the default workflow with three dtypes and the full `decode-4k` profile:

```bash
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k \
/data-aisoft/infinibench-scripts/scripts/run_metax_benchmark.sh
```

Run the reduced development profile:

```bash
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_metax_benchmark.sh
```

Backfill the code versions that were current on a historical date:

```bash
BACKFILL_DATE=2026-09-11 \
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_metax_benchmark.sh
```

Retry a benchmark immediately after a successful build without rebuilding:

```bash
BUILD_INFINIOPS=0 \
BACKFILL_DATE=2026-09-11 \
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_metax_benchmark.sh
```

Only use `BUILD_INFINIOPS=0` when the installed `infini.ops` package was just
built from the requested source revisions and those revisions have not changed.
The script does not verify that the installed wheel and the checked-out source
tree have identical content when compilation is skipped.

## Execution model

One invocation performs these steps:

1. Validate parameters and acquire the machine-level runner lock.
2. Create the canonical operator output directory under
   `/data-aisoft/infinibench_data/ops`.
3. Prepare `InfiniOps` and `InfiniRT` in the container.
4. Select a physical C550 according to the configured resource gate.
5. Build InfiniOps inside the container with the selected InfiniRT source tree.
6. Refresh automatic GPU selection after compilation.
7. Run GPU and Python/runtime preflight checks.
8. Benchmark every requested dtype/profile combination.
9. Export one combined CSV, diagnostics CSV, and Markdown report.
10. Write non-empty completion metadata to `benchmark.done`.

The machine-level lock prevents two runners from simultaneously switching the
shared `/workspace/InfiniOps` and `/workspace/InfiniRT` trees or competing for
the same persistent build cache. If another run is active, the second run exits
with code 2.

## Time and output naming

There are two time concepts:

- `EXECUTION_TIME`: actual execution time. It defaults to the current time and
  can be overridden as `YYYYMMDD-HHMMSS`.
- `BACKFILL_DATE`: optional historical source date in `YYYY-MM-DD` format.

A normal output directory is:

```text
/data-aisoft/infinibench_data/ops/metax-c550/<EXECUTION_TIME>/
```

For example:

```text
/data-aisoft/infinibench_data/ops/metax-c550/20260916-083000/
```

A backfill output directory is:

```text
/data-aisoft/infinibench_data/ops/metax-c550/backfill-<YYYYMMDD>-<EXECUTION_DATE>/
```

For example:

```text
/data-aisoft/infinibench_data/ops/metax-c550/backfill-20260911-20260916/
```

The CSV `date` column is the actual execution date for a normal run and the
backfill target date for a backfill run. `RUN_DATE` and `CSV_DATE` are not
accepted. If a scheduler needs a deterministic output path, set `EXECUTION_TIME`
explicitly before launching the command.

A backfill directory is intentionally deterministic for one target date and one
execution date. A non-empty directory with the same name causes the runner to
exit instead of mixing stale checkpoints into a new attempt. To replace a failed
or superseded attempt in the same fixed directory, set `REUSE_RUN_DIR=1`; this
deletes and recreates only that run directory after the machine lock is held.

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

If a repository does not exist, it is cloned with full history. If it already
exists, the runner requires a clean Git work tree, updates `origin`, fetches
branches and tags, and fast-forwards the configured local branch. A dirty work
tree causes an immediate failure; the runner never discards local changes.

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
`INFINIRT_REF` values take precedence over the date. A commit, tag, or branch is
accepted. Set both refs when an exact repository pair is required.

The selected URL and commits are recorded in `benchmark-launcher.log` and
`benchmark.done`.

## Compilation

Compilation is enabled by default. The runner installs `/workspace/InfiniOps`
inside the container using scikit-build-core. InfiniRT is not installed as a
separate package; InfiniOps CMake imports the configured InfiniRT source tree
and builds the required InfiniRT targets in the same Ninja build.

The persistent build directory is:

```text
/workspace/.cache/infiniops-build
```

The default parallelism is:

```text
CMAKE_BUILD_PARALLEL_LEVEL=64
```

The first full build or a large backfill can take many minutes. A repeat run for
the same revisions normally performs incremental checks and packaging rather
than recompiling every target. Switching source versions or changing CMake
options can cause a substantially larger rebuild.

## GPU selection

Automatic selection is enabled by default. The priority is:

```text
positional GPU argument > PHYSICAL_GPU > automatic selection
```

The default exclusive-device gate requires:

- no existing process on the GPU;
- no `unavailable_reason`;
- utilization no greater than 5%;
- free-memory ratio no less than the configured minimum.

Automatic selection is performed before compilation and refreshed after
compilation. An explicit GPU request is never changed automatically.

For a functional smoke run that may share a device:

```bash
GPU_MAX_UTILIZATION_PERCENT=50 \
GPU_MIN_FREE_MEMORY_PERCENT=50 \
GPU_ALLOW_OCCUPIED=1 \
BUILD_INFINIOPS=0 \
BACKFILL_DATE=2026-09-11 \
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_metax_benchmark.sh
```

`GPU_ALLOW_OCCUPIED=1` allows an existing foreign process and logs it as
`foreign_allowed`; it does not terminate that process. Such a run is useful for
pipeline validation, but its latency data must not be compared with exclusive
benchmark results.

In `mx-smi`, `Memory-Usage` is displayed as used/total memory. For example:

```text
9823/65536 MiB
```

means:

```text
(65536 - 9823) / 65536 ≈ 85% free memory
```

`GPU-Util` is compute utilization and is separate from memory usage.

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

`decode-4k` is the normal coverage profile. `decode-4k-small` uses smaller
shapes and an 18-operator representative set for development and workflow
validation. `decode-8k` and `prefill-4k` are optional shape-specific profiles.

## Parameter reference

| Parameter | Default | Meaning |
|---|---:|---|
| `DTYPES` | `float16,float32,bfloat16` | Dtypes to benchmark in one invocation. |
| `PROFILES` | `decode-4k` | Profiles to benchmark in one invocation. |
| `EXECUTION_TIME` | Current local time | Actual execution time and output label, `YYYYMMDD-HHMMSS`. |
| `BACKFILL_DATE` | empty | Historical source date, `YYYY-MM-DD`. |
| `REUSE_RUN_DIR` | `0` | Allow a deterministic backfill run to replace its existing fixed output directory. |
| `BUILD_INFINIOPS` | `1` | Build and install InfiniOps before benchmarking. |
| `UPDATE_SOURCES` | `1` | Clone/fetch/check out source repositories. |
| `INFINIOPS_REF` | empty | Explicit InfiniOps commit, tag, or branch. |
| `INFINIRT_REF` | empty | Explicit InfiniRT commit, tag, or branch. |
| `INFINIOPS_REPO_URLS` | GitHub plus two proxies | Comma-separated InfiniOps URL fallback chain. |
| `INFINIRT_REPO_URLS` | GitHub plus two proxies | Comma-separated InfiniRT URL fallback chain. |
| `INFINIOPS_BRANCH` | `master` | Configured InfiniOps branch. |
| `INFINIRT_BRANCH` | `master` | Configured InfiniRT branch. |
| `SOURCE_PROBE_TIMEOUT_SECONDS` | `30` | Per-URL reachability probe timeout. |
| `SOURCE_OPERATION_TIMEOUT_SECONDS` | `900` | Clone/fetch operation timeout. |
| `AUTO_SELECT_GPU` | `1` | Automatically select a physical GPU when no explicit index is supplied. |
| `PHYSICAL_GPU` | empty | Explicit physical GPU index. A positional argument takes precedence. |
| `GPU_MAX_UTILIZATION_PERCENT` | `5` | Maximum allowed compute utilization. |
| `GPU_MIN_FREE_MEMORY_PERCENT` | `0` | Minimum required free VRAM percentage. |
| `GPU_ALLOW_OCCUPIED` | `0` | Allow an existing process on the selected GPU. |
| `CMAKE_BUILD_PARALLEL_LEVEL` | `64` | Ninja parallel build level. |
| `INFINIOPS_BUILD_DIR` | `/workspace/.cache/infiniops-build` | Persistent scikit-build directory. |
| `CONTAINER` | `baoming_infiniops_metax` | Docker container used for compilation and benchmarks. |
| `DEVICE_ID` | `metax-c550` | Device directory name under the output root. |
| `OUTPUT_ROOT` | `/data-aisoft/infinibench_data/ops` | Host canonical operator output root. |
| `CONTAINER_OUTPUT_ROOT` | Same as `OUTPUT_ROOT` | Output root visible inside the container. |
| `CONTAINER_BENCHMARK_ROOT` | `/data-aisoft/infinibench-scripts` | Shared benchmark tool root inside the container. |
| `INFINIOPS_SOURCE_ROOT` | `/workspace/InfiniOps` | InfiniOps source path inside the container. |
| `INFINIRT_SOURCE_ROOT` | `/workspace/InfiniRT` | InfiniRT source path inside the container. |
| `TESTER` | `baoming` | `tester` value written to the exported CSV. |

Invalid parameter values, a dirty source tree, an unavailable container, another
active runner, or no GPU satisfying the resource gate normally produce exit code
2. A foreign process on the selected GPU in exclusive mode produces exit code 3.
Build, benchmark, and export failures use the underlying command's nonzero exit
code.

## Output files

For an output directory named `$RUN_DIR`, the main files are:

```text
$RUN_DIR/infiniops-metax-c550.csv
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
infiniops-metax-c550.csv
```

The host and container output roots must resolve to the same bind mount. The
runner checks this with a marker created and removed by the container so shared
accounts do not need permission to unlink files in the sticky output directory.

## Monitoring and completion

Watch the main launcher log:

```bash
tail -f "$RUN_DIR/benchmark-launcher.log"
```

Watch completed Ninja build steps:

```bash
docker exec -it baoming_infiniops_metax \
  tail -f /workspace/.cache/infiniops-build/.ninja_log
```

Show a completed/remaining Ninja step snapshot:

```bash
docker exec baoming_infiniops_metax bash -c '
build=/workspace/.cache/infiniops-build
completed=$(awk "BEGIN{n=0} /^[0-9]/{n++} END{print n}" "$build/.ninja_log")
dry=$(ninja -C "$build" -n 2>/dev/null)
if [[ "$dry" == *"no work to do."* ]]; then
  remaining=0
else
  remaining=$(printf "%s\n" "$dry" | wc -l)
fi
echo "completed_steps=$completed"
echo "remaining_steps=$remaining"
'
```

`benchmark.done` is created empty at startup and is populated only at the end.
Therefore, test existence with:

```bash
test -s "$RUN_DIR/benchmark.done"
```

A successful finished run has:

```text
exit_code=0
foreign_process_seen=0
```

and a generated `infiniops-metax-c550.csv`. The metadata also records the
selected GPU, GPU thresholds, execution time, CSV date, source date, requested
refs, InfiniOps commit, and InfiniRT commit.

## Troubleshooting

### GitHub connection times out

The runner automatically tries the configured proxy fallback chain. To use a
different internal mirror, set the complete comma-separated URL chain through
`INFINIOPS_REPO_URLS` and `INFINIRT_REPO_URLS`.

### Source preparation refuses to continue

Inspect the two work trees:

```bash
docker exec baoming_infiniops_metax \
  git -c safe.directory=/workspace/InfiniOps -C /workspace/InfiniOps status --short
docker exec baoming_infiniops_metax \
  git -c safe.directory=/workspace/InfiniRT -C /workspace/InfiniRT status --short
```

Commit, stash, or remove local changes in the container before retrying. The
runner intentionally avoids destructive cleanup.

### No GPU satisfies the resource gate

Wait for an exclusive GPU for comparable latency data. For a smoke run only,
raise `GPU_MAX_UTILIZATION_PERCENT`, raise or lower
`GPU_MIN_FREE_MEMORY_PERCENT` as needed, and set `GPU_ALLOW_OCCUPIED=1`.
Remember that shared-device latency is not comparable.

### A second invocation exits immediately

Another MetaX runner is active and holds:

```text
/data-aisoft/infinibench_data/ops/metax-c550/.infinibench-runner.lock
```

Wait for the active run to finish instead of removing the lock manually.

### Permission denied in the output directory

The runner creates and removes its path-check marker inside the container. If
permission errors remain, verify that `/data-aisoft` is mounted at the same path
in the container and that the invoking account may execute Docker and `mx-smi`.
