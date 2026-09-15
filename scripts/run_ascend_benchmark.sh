#!/usr/bin/env bash
set -euo pipefail

: "${CONTAINER:?set CONTAINER to a healthy Ascend development container}"

physical_npu="${1:-${PHYSICAL_NPU:-}}"
if [[ -z "$physical_npu" ]]; then
  echo "usage: CONTAINER=<name> $0 <physical-npu-index>" >&2
  exit 2
fi

run_date="${RUN_DATE:-$(date +%Y-%m-%d)}"
tester="${TESTER:-baoming}"
build_infiniops="${BUILD_INFINIOPS:-0}"
build_ascend_custom="${BUILD_ASCEND_CUSTOM:-ON}"
build_with_torch="${BUILD_WITH_TORCH:-OFF}"
docker_timeout_seconds="${DOCKER_TIMEOUT_SECONDS:-15}"

host_tools_root="${BENCHMARK_TOOLS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
host_deployment_root="$(cd "$host_tools_root/.." && pwd)"
output_host="${OUTPUT_ROOT:-$host_deployment_root/InfiniOps_benchmark_results}/$run_date"

# A future benchmark container should mount /data/shared/baoming at /workspace.
container_tools_root="${CONTAINER_BENCHMARK_ROOT:-/workspace/InfiniOps_benchmark_tools}"
source_root="${INFINIOPS_SOURCE_ROOT:-/workspace/InfiniOps}"
infinirt_source_root="${INFINIRT_SOURCE_ROOT:-/workspace/InfiniRT}"
output_container_root="${CONTAINER_OUTPUT_ROOT:-/workspace/InfiniOps_benchmark_results}"
output_container="$output_container_root/$run_date"
harness="$container_tools_root/scripts/full_operator_benchmark.py"
exporter="$container_tools_root/scripts/export_full_operator_report.py"
launcher_log="$output_host/benchmark-launcher.log"
done_file="$output_host/benchmark.done"
csv_date="${run_date//-//}"

target_ops=abs,add,addmm,amax,amin,argmax,bitwise_and,bitwise_not,bitwise_or,bmm,cat,ceil,clamp,cos,cumsum,div,eq,exp,floor,gather,ge,gelu,gt,hardtanh,index_select,le,log,log_softmax,lt,matmul,maximum,mean,minimum,mm,mul,ne,neg,pow,reciprocal,relu,round,rsqrt,sigmoid,sign,silu,sin,softmax,sort,sqrt,sub,sum,tanh,topk
decode_4k_ops="${ASCEND_DECODE_4K_OPS:-all}"
decode_8k_ops="${ASCEND_DECODE_8K_OPS:-$target_ops}"
prefill_4k_ops="${ASCEND_PREFILL_4K_OPS:-$target_ops}"

mkdir -p "$output_host"
: >"$launcher_log"
: >"$done_file"

docker_checked() {
  timeout "$docker_timeout_seconds" docker "$@"
}

if ! docker_checked inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "container is missing or Docker did not respond within ${docker_timeout_seconds}s: $CONTAINER" >&2
  exit 2
fi
if [[ "$(docker_checked inspect -f '{{.State.Running}}' "$CONTAINER")" != "true" ]]; then
  echo "container is not running: $CONTAINER" >&2
  exit 2
fi

# infini/ops.so links against libtorch_python.so through libtorch. Docker does
# not apply an ELF RUNPATH recursively to transitive dependencies, so preserve
# the image's CANN paths and explicitly add the installed Torch library paths.
container_ld_library_path="$({
  docker_checked inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER" |
    sed -n 's/^LD_LIBRARY_PATH=//p'
} | tail -n 1)"
torch_library_path="$(docker_checked exec "$CONTAINER" python3 -c '
import importlib.util
from pathlib import Path

paths = []
for package in ("torch", "torch_npu"):
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise SystemExit(f"cannot locate {package}")
    paths.append(str(Path(spec.origin).resolve().parent / "lib"))
print(":".join(paths))
')"
runtime_ld_library_path="$torch_library_path"
if [[ -n "$container_ld_library_path" ]]; then
  runtime_ld_library_path+=":$container_ld_library_path"
fi

docker_checked exec "$CONTAINER" test -d "$source_root"
docker_checked exec "$CONTAINER" test -d "$infinirt_source_root"
docker_checked exec "$CONTAINER" test -f "$harness"
docker_checked exec "$CONTAINER" test -f "$exporter"
IFS=: read -r -a runtime_library_dirs <<<"$torch_library_path"
for runtime_library_dir in "${runtime_library_dirs[@]}"; do
  docker_checked exec "$CONTAINER" test -d "$runtime_library_dir"
done
docker_checked exec "$CONTAINER" mkdir -p "$output_container"

# Both paths must resolve to the same bind mount. Otherwise the launcher logs
# and benchmark payloads would be split across different host directories.
output_path_marker=".benchmark-output-path-check-$$"
: >"$output_host/$output_path_marker"
if ! docker_checked exec "$CONTAINER" test -f "$output_container/$output_path_marker"; then
  rm -f -- "$output_host/$output_path_marker"
  echo "OUTPUT_ROOT and CONTAINER_OUTPUT_ROOT do not point to the same mounted directory" >&2
  echo "host: $output_host" >&2
  echo "container: $output_container" >&2
  exit 2
fi
rm -f -- "$output_host/$output_path_marker"

if [[ "$build_infiniops" == "1" ]]; then
  build_command=(
    python3 -m pip install
    --no-build-isolation
    --no-deps
    --force-reinstall
    "$source_root"
    --config-settings=cmake.define.AUTO_DETECT_DEVICES=OFF
    --config-settings=cmake.define.AUTO_DETECT_BACKENDS=OFF
    --config-settings=cmake.define.WITH_CPU=ON
    --config-settings=cmake.define.WITH_ASCEND=ON
    "--config-settings=cmake.define.WITH_TORCH=$build_with_torch"
    --config-settings=cmake.define.WITH_LINKED=OFF
    "--config-settings=cmake.define.INFINI_RT_SOURCE_DIR=$infinirt_source_root"
    "--config-settings=cmake.define.BUILD_ASCEND_CUSTOM=$build_ascend_custom"
  )
  if [[ -n "${SOC_VERSION:-}" ]]; then
    build_command+=("--config-settings=cmake.define.SOC_VERSION=$SOC_VERSION")
  fi
  docker exec "$CONTAINER" "${build_command[@]}" 2>&1 | tee -a "$launcher_log"
fi

metadata="${INFINIOPS_METADATA:-$source_root/generated/torch_ops_metadata.json}"
docker_checked exec "$CONTAINER" test -f "$metadata"
sha="$(
  docker_checked exec "$CONTAINER" \
    git -c "safe.directory=$source_root" -C "$source_root" rev-parse HEAD
)"

{
  echo "start=$(date --iso-8601=seconds)"
  echo "container=$CONTAINER"
  echo "physical_npu=$physical_npu"
  echo "run_date=$run_date"
  echo "source_root=$source_root"
  echo "infinirt_source_root=$infinirt_source_root"
  echo "sha=$sha"
  echo "metadata=$metadata"
  echo "build_infiniops=$build_infiniops"
  echo "build_ascend_custom=$build_ascend_custom"
  echo "build_with_torch=$build_with_torch"
  echo "soc_version=${SOC_VERSION:-auto-detect}"
  echo "profiles=decode-4k,decode-8k,prefill-4k"
  echo "decode_4k_ops=$decode_4k_ops"
  echo "decode_8k_ops=$decode_8k_ops"
  echo "prefill_4k_ops=$prefill_4k_ops"
  echo "torch_library_path=$torch_library_path"
  echo "=== host preflight ==="
  npu-smi info || true
  echo "=== container preflight ==="
  docker exec \
    -e ASCEND_RT_VISIBLE_DEVICES="$physical_npu" \
    -e LD_LIBRARY_PATH="$runtime_ld_library_path" \
    "$CONTAINER" python3 -c \
    "import torch, torch_npu, infini.ops; assert torch.npu.is_available(); print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_count', torch.npu.device_count())"
} >>"$launcher_log" 2>&1

run_profile() {
  local profile="$1"
  local selected_ops="${2:-}"
  local output="$output_container/$profile"
  local -a command=(
    python3 "$harness"
    --metadata "$metadata"
    --output-dir "$output"
    --device npu
    --dtype float16
    --sha "$sha"
    --min-run-time 0.02
    --timeout 20
  )
  if [[ -n "$selected_ops" && "$selected_ops" != "all" ]]; then
    command+=(--ops "$selected_ops")
  fi
  {
    echo "=== profile=$profile start=$(date --iso-8601=seconds) ==="
    printf 'command='
    printf '%q ' "${command[@]}"
    echo
  } >>"$launcher_log"
  docker exec \
    -e ASCEND_RT_VISIBLE_DEVICES="$physical_npu" \
    -e INFINIOPS_BENCHMARK_PROFILE="$profile" \
    -e LD_LIBRARY_PATH="$runtime_ld_library_path" \
    "$CONTAINER" "${command[@]}" 2>&1 | tee -a "$launcher_log"
  return "${PIPESTATUS[0]}"
}

exit_code=0
run_profile decode-4k "$decode_4k_ops" || exit_code=$?
if [[ "$exit_code" -eq 0 ]]; then
  run_profile decode-8k "$decode_8k_ops" || exit_code=$?
fi
if [[ "$exit_code" -eq 0 ]]; then
  run_profile prefill-4k "$prefill_4k_ops" || exit_code=$?
fi

if [[ "$exit_code" -eq 0 ]]; then
  docker exec \
    -e LD_LIBRARY_PATH="$runtime_ld_library_path" \
    "$CONTAINER" python3 "$exporter" \
    "$output_container/decode-4k/results.json" \
    "$output_container/decode-8k/results.json" \
    "$output_container/prefill-4k/results.json" \
    --output "$output_container/infiniops-ascend-910b.csv" \
    --diagnostics "$output_container/diagnostics.csv" \
    --report "$output_container/report.md" \
    --platform ascend \
    --date "$csv_date" \
    --tester "$tester" 2>&1 | tee -a "$launcher_log"
  exit_code=${PIPESTATUS[0]}
fi

{
  echo "exit_code=$exit_code"
  echo "end=$(date --iso-8601=seconds)"
  echo "container=$CONTAINER"
  echo "physical_npu=$physical_npu"
  echo "sha=$sha"
} >"$done_file"

exit "$exit_code"
