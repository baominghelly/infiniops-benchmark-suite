#!/usr/bin/env bash
set -euo pipefail

container="${CONTAINER:-baoming_infiniops_metax}"
requested_gpu="${1:-${PHYSICAL_GPU:-}}"
auto_select_gpu="${AUTO_SELECT_GPU:-1}"
physical_gpu=""
gpu_selection=""
gpu_candidates=""
gpu_busy=""
gpu_rejected=""

run_date="${RUN_DATE:-$(date +%Y-%m-%d)}"
tester="${TESTER:-baoming}"
build_infiniops="${BUILD_INFINIOPS:-1}"
device_id="${DEVICE_ID:-metax-c550}"
output_root="${OUTPUT_ROOT:-/data-aisoft/infinibench_data}"
dtypes_input="${DTYPES:-float16,float32,bfloat16}"
read -r -a dtypes <<<"${dtypes_input//,/ }"
for dtype in "${dtypes[@]}"; do
  case "$dtype" in
    float16|float32|bfloat16) ;;
    *)
      echo "invalid DTYPES entry: $dtype; expected float16, float32, or bfloat16" >&2
      exit 2
      ;;
  esac
done
profiles_input="${PROFILES:-decode-4k}"
read -r -a profiles <<<"${profiles_input//,/ }"
for profile in "${profiles[@]}"; do
  case "$profile" in
    decode-4k|decode-4k-small|decode-8k|prefill-4k) ;;
    *)
      echo "invalid PROFILES entry: $profile; expected decode-4k, decode-4k-small, decode-8k, or prefill-4k" >&2
      exit 2
      ;;
  esac
done

host_tools_root="${BENCHMARK_TOOLS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
output_host="$output_root/$device_id/$run_date"
container_tools_root="${CONTAINER_BENCHMARK_ROOT:-/workspace/InfiniOps_benchmark_tools}"
source_root="${INFINIOPS_SOURCE_ROOT:-/workspace/InfiniOps}"
infinirt_source_root="${INFINIRT_SOURCE_ROOT:-/workspace/InfiniRT}"
build_dir="${INFINIOPS_BUILD_DIR:-/workspace/.cache/infiniops-build}"
cmake_build_parallel_level="${CMAKE_BUILD_PARALLEL_LEVEL:-64}"
output_container_root="${CONTAINER_OUTPUT_ROOT:-$output_root}"
output_container="$output_container_root/$device_id/$run_date"
harness="$container_tools_root/scripts/full_operator_benchmark.py"
exporter="$container_tools_root/scripts/export_full_operator_report.py"
launcher_log="$output_host/benchmark-launcher.log"
monitor_log="$output_host/benchmark-monitor.log"
foreign_flag="$output_host/foreign-process.flag"
done_file="$output_host/benchmark.done"
csv_date="${CSV_DATE:-${run_date//-//}}"

target_ops=abs,add,addmm,amax,amin,argmax,bitwise_and,bitwise_not,bitwise_or,bmm,cat,ceil,clamp,cos,cumsum,div,eq,exp,floor,gather,ge,gelu,gt,hardtanh,index_select,le,log,log_softmax,lt,matmul,maximum,mean,minimum,mm,mul,ne,neg,pow,reciprocal,relu,round,rsqrt,sigmoid,sign,silu,sin,softmax,sort,sqrt,sub,sum,tanh,topk
small_ops=abs,add,avg_pool2d,bmm,gather,fft_fft,gelu,matmul,mm,mul,relu,sigmoid,silu,softmax,special_gammaln,sparse_sampled_addmm,sum,where
decode_4k_ops="${METAX_DECODE_4K_OPS:-all}"
decode_4k_small_ops="${METAX_DECODE_4K_SMALL_OPS:-$small_ops}"
decode_8k_ops="${METAX_DECODE_8K_OPS:-$target_ops}"
prefill_4k_ops="${METAX_PREFILL_4K_OPS:-$target_ops}"

mkdir -p "$output_host"
: >"$launcher_log"
: >"$monitor_log"
: >"$foreign_flag"
: >"$done_file"

if ! docker inspect "$container" >/dev/null 2>&1; then
  echo "container not found: $container" >&2
  exit 2
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" != "true" ]]; then
  echo "container is not running: $container" >&2
  exit 2
fi
container_id="$(docker inspect -f '{{.Id}}' "$container")"

docker exec "$container" test -d "$source_root"
docker exec "$container" test -d "$infinirt_source_root"
docker exec "$container" test -f "$harness"
docker exec "$container" test -f "$exporter"
docker exec "$container" mkdir -p "$output_container"

output_path_marker=".benchmark-output-path-check-$$"
: >"$output_host/$output_path_marker"
if ! docker exec "$container" test -f "$output_container/$output_path_marker"; then
  rm -f -- "$output_host/$output_path_marker"
  echo "OUTPUT_ROOT and CONTAINER_OUTPUT_ROOT do not point to the same mounted directory" >&2
  exit 2
fi
rm -f -- "$output_host/$output_path_marker"

select_idle_gpu() {
  local gpu_json process_table selection_output
  if ! gpu_json="$(mx-smi -j 2>&1)"; then
    echo "mx-smi -j failed:" >&2
    printf '%s\n' "$gpu_json" >&2
    return 2
  fi
  if ! process_table="$(mx-smi --show-process 2>&1)"; then
    echo "mx-smi --show-process failed:" >&2
    printf '%s\n' "$process_table" >&2
    return 2
  fi

  selection_output="$(
    {
      printf '%s\n' "$gpu_json"
      printf '\n__MX_SMI_PROCESS_TABLE__\n'
      printf '%s\n' "$process_table"
    } | python3 -c '
import json
import sys

raw = sys.stdin.read()
marker = "__MX_SMI_PROCESS_TABLE__"
if marker not in raw:
    raise SystemExit("mx-smi output did not contain the process-table marker")
json_text, process_text = raw.split(marker, 1)
json_start = json_text.find("{")
if json_start < 0:
    raise SystemExit("mx-smi -j output did not contain JSON")
payload = json.loads(json_text[json_start:])

devices = []
for value in payload.values():
    if isinstance(value, dict) and isinstance(value.get("device_id"), int):
        devices.append(value)
if not devices:
    raise SystemExit("mx-smi -j output did not contain any MetaX devices")

busy = set()
for line in process_text.splitlines():
    columns = [column.strip() for column in line.split("|")]
    if len(columns) >= 4 and columns[1].isdigit() and columns[2].isdigit():
        busy.add(int(columns[1]))

eligible = []
rejected = []
for device in sorted(devices, key=lambda item: item["device_id"]):
    device_id = device["device_id"]
    utilization = device.get("utilization")
    unavailable_reason = str(device.get("unavailable_reason") or "")
    reasons = []
    if device_id in busy:
        reasons.append("busy")
    if unavailable_reason:
        reasons.append(f"unavailable={unavailable_reason}")
    try:
        utilization_value = int(utilization)
    except (TypeError, ValueError):
        reasons.append(f"invalid_utilization={utilization}")
    else:
        if utilization_value > 5:
            reasons.append(f"utilization={utilization_value}")
    if reasons:
        rejected.append(str(device_id) + "(" + ",".join(reasons) + ")")
    else:
        eligible.append(device_id)

if not eligible:
    raise SystemExit(
        "no idle MetaX GPU found; rejected: "
        + (", ".join(rejected) if rejected else "unknown")
    )

selected = eligible[0]
print(selected)
print(",".join(str(item) for item in eligible))
print(",".join(str(item) for item in sorted(busy)) or "-")
print(";".join(rejected) or "-")
'
  )" || return 2

  mapfile -t selection_rows <<<"$selection_output"
  if [[ "${#selection_rows[@]}" -ne 4 ]]; then
    echo "unexpected idle-GPU selector output: $selection_output" >&2
    return 2
  fi
  physical_gpu="${selection_rows[0]}"
  gpu_candidates="${selection_rows[1]}"
  gpu_busy="${selection_rows[2]}"
  gpu_rejected="${selection_rows[3]}"
  gpu_selection="auto"
}

if [[ -n "$requested_gpu" ]]; then
  if [[ ! "$requested_gpu" =~ ^[0-9]+$ ]]; then
    echo "invalid physical GPU index: $requested_gpu" >&2
    exit 2
  fi
  physical_gpu="$requested_gpu"
  gpu_selection="explicit"
elif [[ "$auto_select_gpu" == "1" ]]; then
  if ! select_idle_gpu; then
    exit 2
  fi
else
  echo "usage: [CONTAINER=<name>] [AUTO_SELECT_GPU=0] [PHYSICAL_GPU=<index>] $0 [<physical-gpu-index>]" >&2
  exit 2
fi

{
  echo "gpu_selection=$gpu_selection"
  echo "physical_gpu=$physical_gpu"
  echo "gpu_candidates=$gpu_candidates"
  echo "gpu_busy=$gpu_busy"
  echo "gpu_rejected=$gpu_rejected"
} >>"$launcher_log"

if [[ "$build_infiniops" == "1" ]]; then
  docker exec "$container" mkdir -p "$build_dir"
  {
    echo "=== build start=$(date --iso-8601=seconds) ==="
    echo "build_dir=$build_dir"
    echo "cmake_build_parallel_level=$cmake_build_parallel_level"
  } >>"$launcher_log"
  build_exit=0
  docker exec \
    -e CMAKE_BUILD_PARALLEL_LEVEL="$cmake_build_parallel_level" \
    "$container" python3 -m pip install \
      --no-build-isolation \
      --no-deps \
      --force-reinstall \
      "$source_root" \
      "--config-settings=build-dir=$build_dir" \
      --config-settings=cmake.define.AUTO_DETECT_DEVICES=OFF \
      --config-settings=cmake.define.AUTO_DETECT_BACKENDS=OFF \
      --config-settings=cmake.define.WITH_CPU=ON \
      --config-settings=cmake.define.WITH_METAX=ON \
      --config-settings=cmake.define.WITH_TORCH=ON \
      --config-settings=cmake.define.WITH_LINKED=OFF \
      "--config-settings=cmake.define.INFINI_RT_SOURCE_DIR=$infinirt_source_root" \
      "--config-settings=cmake.define.CMAKE_C_COMPILER=$source_root/scripts/mxcc_wrapper.sh" \
      "--config-settings=cmake.define.CMAKE_CXX_COMPILER=$source_root/scripts/mxcc_wrapper.sh" \
      2>&1 | tee -a "$launcher_log" || build_exit=$?
  if [[ "$build_exit" -ne 0 ]]; then
    echo "build_exit=$build_exit" | tee -a "$launcher_log"
    exit "$build_exit"
  fi
  echo "=== build end=$(date --iso-8601=seconds) ===" >>"$launcher_log"
fi

# Compilation can take long enough for the initially selected device to become
# occupied. Refresh automatic selection immediately before benchmark preflight.
# An explicit GPU request is never changed automatically.
if [[ "$gpu_selection" == "auto" ]]; then
  previous_gpu="$physical_gpu"
  if ! select_idle_gpu; then
    exit 2
  fi
  {
    echo "gpu_reselection=auto"
    echo "previous_gpu=$previous_gpu"
    echo "physical_gpu=$physical_gpu"
    echo "gpu_candidates=$gpu_candidates"
    echo "gpu_busy=$gpu_busy"
    echo "gpu_rejected=$gpu_rejected"
  } >>"$launcher_log"
fi

metadata="${INFINIOPS_METADATA:-$source_root/generated/torch_ops_metadata.json}"
docker exec "$container" test -f "$metadata"
sha="$(docker exec "$container" git -c "safe.directory=$source_root" -C "$source_root" rev-parse HEAD)"
infinirt_sha="$(docker exec "$container" git -c "safe.directory=$infinirt_source_root" -C "$infinirt_source_root" rev-parse HEAD)"

process_snapshot="$(mx-smi -i "$physical_gpu" --show-process 2>&1)"
if ! grep -q "no process found" <<<"$process_snapshot"; then
  printf '%s\n' "$process_snapshot" >&2
  echo "physical MetaX GPU $physical_gpu has a running process" >&2
  exit 2
fi
utilization="$(mx-smi -i "$physical_gpu" -j | sed -n '/^[[:space:]]*{/,$p' | python3 -c '
import json, sys
payload = json.load(sys.stdin)
for value in payload.values():
    if isinstance(value, dict) and value.get("device_id") == int(sys.argv[1]):
        print(value.get("utilization"))
        break
else:
    raise SystemExit(f"MetaX GPU {sys.argv[1]} was not found")
' "$physical_gpu")"
if [[ -z "$utilization" || "$utilization" =~ [^0-9] || "$utilization" -gt 5 ]]; then
  echo "physical MetaX GPU $physical_gpu utilization is not idle: $utilization" >&2
  exit 2
fi

{
  echo "start=$(date --iso-8601=seconds)"
  echo "container=$container"
  echo "container_id=$container_id"
  echo "physical_gpu=$physical_gpu"
  echo "run_date=$run_date"
  echo "source_root=$source_root"
  echo "sha=$sha"
  echo "infinirt_sha=$infinirt_sha"
  echo "metadata=$metadata"
  echo "build_infiniops=$build_infiniops"
  echo "dtypes=$dtypes_input"
  echo "profiles=$profiles_input"
  echo "decode_4k_ops=$decode_4k_ops"
  echo "decode_4k_small_ops=$decode_4k_small_ops"
  echo "decode_8k_ops=$decode_8k_ops"
  echo "prefill_4k_ops=$prefill_4k_ops"
  echo "=== host preflight ==="
  mx-smi -i "$physical_gpu"
  printf '%s\n' "$process_snapshot"
  echo "=== container preflight ==="
  docker exec -e CUDA_VISIBLE_DEVICES="$physical_gpu" "$container" python3 -c \
    "import torch, infini.ops; assert torch.cuda.is_available(); print('torch', torch.__version__); print('cuda_count', torch.cuda.device_count()); print('device', torch.cuda.get_device_name(0))"
} >>"$launcher_log" 2>&1

monitor() {
  local snapshot pids pid cgroup
  while [[ ! -s "$done_file" ]]; do
    echo "sample=$(date --iso-8601=seconds)"
    if ! snapshot="$(mx-smi -i "$physical_gpu" --show-process 2>&1)"; then
      echo "monitor_error=$snapshot"
      echo "mx-smi-error" >>"$foreign_flag"
      docker exec "$container" pkill -f "$harness" 2>/dev/null || true
      break
    fi
    printf '%s\n' "$snapshot"
    if grep -q "no process found" <<<"$snapshot"; then
      sleep 2
      continue
    fi
    pids="$(awk -F '|' -v gpu="$physical_gpu" '
      $2 ~ /^[[:space:]]*[0-9]+[[:space:]]*$/ && trim($2) == gpu { print trim($3) }
      function trim(text) { gsub(/^[[:space:]]+|[[:space:]]+$/, "", text); return text }
    ' <<<"$snapshot")"
    for pid in $pids; do
      if [[ ! -e "/proc/$pid" ]]; then
        echo "process_owner=exited_before_attribution pid=$pid"
        continue
      fi
      cgroup="$(tr '\n' ' ' <"/proc/$pid/cgroup" 2>/dev/null || true)"
      if [[ "$cgroup" == *"$container_id"* ]]; then
        echo "process_owner=benchmark_container pid=$pid cmdline=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
      else
        echo "process_owner=foreign pid=$pid cgroup=$cgroup cmdline=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
        if [[ ! -s "$foreign_flag" ]]; then
          echo "$pid" >>"$foreign_flag"
          docker exec "$container" pkill -f "$harness" 2>/dev/null || true
        fi
      fi
    done
    sleep 2
  done
}

run_profile() {
  local dtype="$1"
  local profile="$2"
  local selected_ops="${3:-}"
  local output="$output_container/$dtype/$profile"
  local -a command=(
    python3 "$harness"
    --metadata "$metadata"
    --output-dir "$output"
    --device cuda
    --dtype "$dtype"
    --sha "$sha"
    --min-run-time 0.02
    --timeout 20
  )
  if [[ -n "$selected_ops" && "$selected_ops" != "all" ]]; then
    command+=(--ops "$selected_ops")
  fi
  {
    echo "=== dtype=$dtype profile=$profile start=$(date --iso-8601=seconds) ==="
    printf 'command='
    printf '%q ' "${command[@]}"
    echo
  } >>"$launcher_log"
  docker exec \
    -e CUDA_VISIBLE_DEVICES="$physical_gpu" \
    -e INFINIOPS_BENCHMARK_PROFILE="$profile" \
    "$container" "${command[@]}" 2>&1 | tee -a "$launcher_log"
  return "${PIPESTATUS[0]}"
}

monitor >>"$monitor_log" 2>&1 &
monitor_pid=$!

exit_code=0
exporter_sources=()
for dtype in "${dtypes[@]}"; do
  echo "=== dtype=$dtype start=$(date --iso-8601=seconds) ===" >>"$launcher_log"
  for profile in "${profiles[@]}"; do
    case "$profile" in
      decode-4k) selected_ops="$decode_4k_ops" ;;
      decode-4k-small) selected_ops="$decode_4k_small_ops" ;;
      decode-8k) selected_ops="$decode_8k_ops" ;;
      prefill-4k) selected_ops="$prefill_4k_ops" ;;
    esac
    run_profile "$dtype" "$profile" "$selected_ops" || exit_code=$?
    result_path="$output_container/$dtype/$profile/results.json"
    if docker exec "$container" test -f "$result_path"; then
      exporter_sources+=("$result_path")
    else
      echo "missing_result=$result_path" | tee -a "$launcher_log"
      exit_code=1
    fi
  done
  echo "=== dtype=$dtype end=$(date --iso-8601=seconds) ===" >>"$launcher_log"
done

if [[ "${#exporter_sources[@]}" -gt 0 ]]; then
  docker exec "$container" python3 "$exporter" \
    "${exporter_sources[@]}" \
    --output "$output_container/infiniops-metax-c550.csv" \
    --diagnostics "$output_container/diagnostics.csv" \
    --report "$output_container/report.md" \
    --platform metax \
    --date "$csv_date" \
    --tester "$tester" 2>&1 | tee -a "$launcher_log"
  exporter_exit=${PIPESTATUS[0]}
  if [[ "$exporter_exit" -ne 0 ]]; then
    exit_code="$exporter_exit"
  fi
else
  echo "no result JSON files were available for export" | tee -a "$launcher_log"
  exit_code=1
fi

{
  echo "=== post-run mx-smi ==="
  mx-smi -i "$physical_gpu"
  echo "end=$(date --iso-8601=seconds)"
} >>"$launcher_log" 2>&1

if [[ -s "$foreign_flag" ]]; then
  foreign_process_seen=1
else
  foreign_process_seen=0
fi
{
  echo "exit_code=$exit_code"
  echo "foreign_process_seen=$foreign_process_seen"
  echo "end=$(date --iso-8601=seconds)"
  echo "container=$container"
  echo "physical_gpu=$physical_gpu"
  echo "sha=$sha"
  echo "infinirt_sha=$infinirt_sha"
} >"$done_file"

wait "$monitor_pid"
if [[ "$foreign_process_seen" != 0 ]]; then
  exit 3
fi
exit "$exit_code"
