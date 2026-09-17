# Unified benchmark runner guide

`scripts/run_benchmark.sh` is the single public entry point for all supported
accelerator platforms. It selects a platform adapter from the requested device,
applies the shared operator-data delivery contract, and forwards all remaining
arguments to that platform's runner.

The deployed shared entry point is:

```text
/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh
```

## Device selection

Pass the device as the first argument:

```bash
/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh metax-c550
```

Or set `DEVICE` in the environment:

```bash
DEVICE=metax-c550 \
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k \
/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh
```

Additional arguments are forwarded to the selected platform runner. Therefore
an explicit device index can still be passed after the device:

```bash
/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh nvidia-a100 3
```

If `DEVICE` is set, the first non-option argument is not consumed as a device
and is forwarded directly:

```bash
DEVICE=nvidia-a100 \
/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh 3
```

List supported devices:

```bash
/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh --list
```

Resolve and print the selected adapter and output contract without launching a
benchmark:

```bash
/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh --print metax-c550
```

## Supported devices

| Canonical device | Accepted aliases | Platform runner | Output directory |
|---|---|---|---|
| `nvidia-a100` | `nvidia`, `a100` | `run_a100_benchmark.sh` | `/data-aisoft/infinibench_data/ops/nvidia-a100/` |
| `metax-c550` | `metax`, `c550` | `run_metax_benchmark.sh` | `/data-aisoft/infinibench_data/ops/metax-c550/` |
| `ascend-910c` | `ascend`, `910c` | `run_ascend_benchmark.sh` | `/data-aisoft/infinibench_data/ops/ascend-910c/` |
| `moore-s5000` | `moore` | `run_moore_benchmark.sh` | `/data-aisoft/infinibench_data/ops/moore-s5000/` |
| `iluvatar-tg150` | `iluvatar`, `tianshu` | `run_iluvatar_benchmark.sh` | `/data-aisoft/infinibench_data/ops/iluvatar-tg150/` |
| `cambricon-mlu590` | `cambricon`, `mlu` | `run_cambricon_benchmark.sh` | `/data-aisoft/infinibench_data/ops/cambricon-mlu590/` |
| `hygon-bw1000` | `hygon`, `dcu` | `run_hygon_benchmark.sh` | `/data-aisoft/infinibench_data/ops/hygon-bw1000/` |

The shared entry point exports the canonical `DEVICE_ID` and:

```text
OUTPUT_ROOT=/data-aisoft/infinibench_data/ops
```

Platform runners own their container names, source roots, build caches, GPU
monitoring commands, runtime preflight, and compilation flags. This keeps one
stable public entry point while avoiding an unsafe monolithic script that mixes
seven incompatible hardware toolchains.

## Common workflow

Normal run:

```bash
DEVICE=moore-s5000 \
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k \
/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh
```

Backfill run:

```bash
DEVICE=moore-s5000 \
BACKFILL_DATE=2026-09-04 \
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k \
/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh
```

For all current runners, a normal output directory is:

```text
/data-aisoft/infinibench_data/ops/<device>/<execution-date>/
```

A backfill output directory is:

```text
/data-aisoft/infinibench_data/ops/<device>/backfill-<target-date>-<execution-date>/
```

Directory names do not include the time of day. The complete
`EXECUTION_TIME=YYYYMMDD-HHMMSS` remains in the runner logs and `benchmark.done`
metadata. A non-empty deterministic run directory is rejected by default;
pass `REUSE_RUN_DIR=1` only when replacing a failed or superseded attempt.

## Why platform adapters remain

The unified entry point centralizes device selection and delivery paths, but
hardware execution remains in platform adapters because the platforms use
different container images, monitoring tools, source layouts, compilers, runtime
 APIs, and build caches. For example:

- NVIDIA uses `nvidia-smi`;
- MetaX uses `mx-smi`;
- Ascend uses NPU tooling;
- Moore uses `mthreads-gmi`;
- Hygon uses ROCm/Hygon tooling;
- Cambricon uses MLU tooling;
- Iluvatar/Tianshu uses its deployed stack.

Keeping these adapters separate avoids accidental cross-platform breakage while
still giving agents and frontend workflows one command to call.
