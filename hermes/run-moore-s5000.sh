#!/usr/bin/env bash
set -Eeuo pipefail

export DEVICE="${BENCHMARK_DEVICE:-moore-s5000}"
runner="${BENCHMARK_RUNNER:-/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh}"
test -x "$runner"
exec "$runner"
