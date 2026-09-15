#!/usr/bin/env bash
set -euo pipefail

container="${CONTAINER:-infinicore-cambricon-main-benchmark}"
physical_mlu="${1:-${PHYSICAL_MLU:-0}}"
run_date="${RUN_DATE:-$(date +%Y-%m-%d)}"
tester="${TESTER:-baoming}"
build_infiniops="${BUILD_INFINIOPS:-0}"

host_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
container_root="${CONTAINER_BENCHMARK_ROOT:-/workspace/main-stack-20260904/operator-benchmark}"
source_root="${INFINIOPS_SOURCE_ROOT:-/workspace/main-stack-20260904/src/InfiniCore/submodules/InfiniOps}"
output_host="${OUTPUT_ROOT:-$host_root/results}/$run_date"
output_container="$container_root/results/$run_date"
harness="$container_root/scripts/full_operator_benchmark.py"
exporter="$container_root/scripts/export_full_operator_report.py"
launcher_log="$output_host/benchmark-launcher.log"
done_file="$output_host/benchmark.done"
csv_date="${run_date//-//}"

target_ops=abs,add,addmm,amax,amin,argmax,bitwise_and,bitwise_not,bitwise_or,bmm,cat,ceil,clamp,cos,cumsum,div,eq,exp,floor,gather,ge,gelu,gt,hardtanh,index_select,le,log,log_softmax,lt,matmul,maximum,mean,minimum,mm,mul,ne,neg,pow,reciprocal,relu,round,rsqrt,sigmoid,sign,silu,sin,softmax,sort,sqrt,sub,sum,tanh,topk

mkdir -p "$output_host"
: >"$launcher_log"
: >"$done_file"

if ! docker inspect "$container" >/dev/null 2>&1; then
  echo "container not found: $container" >&2
  exit 2
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" != "true" ]]; then
  echo "container is not running: $container" >&2
  exit 2
fi

if [[ "$build_infiniops" == "1" ]]; then
  docker exec \
    -e CMAKE_ARGS="-DWITH_CPU=ON -DWITH_CAMBRICON=ON -DWITH_TORCH=ON -DWITH_LINKED=OFF" \
    "$container" python3 -m pip install --no-build-isolation --force-reinstall "$source_root" \
    2>&1 | tee -a "$launcher_log"
fi

metadata="$(docker exec "$container" python3 -c \
  "from pathlib import Path; import infini; print(Path(infini.__file__).resolve().parent / 'torch_ops_metadata.json')")"

{
  echo "start=$(date --iso-8601=seconds)"
  echo "container=$container"
  echo "physical_mlu=$physical_mlu"
  echo "run_date=$run_date"
  echo "source_root=$source_root"
  echo "metadata=$metadata"
  echo "build_infiniops=$build_infiniops"
  echo "profiles=decode-4k(full),decode-8k(targeted),prefill-4k(targeted)"
  echo "=== preflight ==="
  cnmon info || true
  docker exec -e MLU_VISIBLE_DEVICES="$physical_mlu" "$container" python3 -c \
    "import torch, torch_mlu, infini.ops; assert torch.mlu.is_available(); print('mlu_count', torch.mlu.device_count())"
} >>"$launcher_log" 2>&1

run_profile() {
  local profile="$1"
  local selected_ops="${2:-}"
  local output="$output_container/$profile"
  local -a command=(
    python3 "$harness"
    --metadata "$metadata"
    --output-dir "$output"
    --device mlu
    --dtype float16
    --min-run-time 0.02
    --timeout 20
  )
  if [[ -n "$selected_ops" ]]; then
    command+=(--ops "$selected_ops")
  fi
  {
    echo "=== profile=$profile start=$(date --iso-8601=seconds) ==="
    printf 'command='
    printf '%q ' "${command[@]}"
    echo
  } >>"$launcher_log"
  docker exec \
    -e MLU_VISIBLE_DEVICES="$physical_mlu" \
    -e INFINIOPS_BENCHMARK_PROFILE="$profile" \
    "$container" "${command[@]}" 2>&1 | tee -a "$launcher_log"
  return "${PIPESTATUS[0]}"
}

exit_code=0
run_profile decode-4k || exit_code=$?
if [[ "$exit_code" -eq 0 ]]; then
  run_profile decode-8k "$target_ops" || exit_code=$?
fi
if [[ "$exit_code" -eq 0 ]]; then
  run_profile prefill-4k "$target_ops" || exit_code=$?
fi

if [[ "$exit_code" -eq 0 ]]; then
  docker exec "$container" python3 "$exporter" \
    "$output_container/decode-4k/results.json" \
    "$output_container/decode-8k/results.json" \
    "$output_container/prefill-4k/results.json" \
    --output "$output_container/infiniops-cambricon-mlu590.csv" \
    --diagnostics "$output_container/diagnostics.csv" \
    --report "$output_container/report.md" \
    --platform cambricon \
    --date "$csv_date" \
    --tester "$tester" 2>&1 | tee -a "$launcher_log"
  exit_code=${PIPESTATUS[0]}
fi

{
  echo "exit_code=$exit_code"
  echo "end=$(date --iso-8601=seconds)"
  echo "container=$container"
  echo "physical_mlu=$physical_mlu"
} >"$done_file"

exit "$exit_code"
