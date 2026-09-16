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

execution_time="${EXECUTION_TIME:-$(date +%Y%m%d-%H%M%S)}"
tester="${TESTER:-baoming}"
build_infiniops="${BUILD_INFINIOPS:-1}"
update_sources="${UPDATE_SOURCES:-1}"
backfill_date="${BACKFILL_DATE:-}"
infiniops_ref="${INFINIOPS_REF:-}"
infinirt_ref="${INFINIRT_REF:-}"
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

if [[ -n "${RUN_DATE:-}" || -n "${CSV_DATE:-}" ]]; then
  echo "RUN_DATE and CSV_DATE are no longer accepted; use optional EXECUTION_TIME and/or BACKFILL_DATE" >&2
  exit 2
fi
execution_time_iso="${execution_time:0:4}-${execution_time:4:2}-${execution_time:6:2} ${execution_time:9:2}:${execution_time:11:2}:${execution_time:13:2}"
if [[ ! "$execution_time" =~ ^[0-9]{8}-[0-9]{6}$ ]] || ! date -d "$execution_time_iso" >/dev/null 2>&1; then
  echo "invalid EXECUTION_TIME: $execution_time; expected YYYYMMDD-HHMMSS" >&2
  exit 2
fi
if [[ -n "$backfill_date" ]]; then
  run_date="backfill-${backfill_date//-/}-${execution_time}"
  csv_date="${backfill_date//-//}"
else
  run_date="$execution_time"
  csv_date="${execution_time:0:4}/${execution_time:4:2}/${execution_time:6:2}"
fi

case "$update_sources" in
  0|1) ;;
  *)
    echo "invalid UPDATE_SOURCES: $update_sources; expected 0 or 1" >&2
    exit 2
    ;;
esac
if [[ -n "$backfill_date" ]]; then
  if [[ ! "$backfill_date" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || ! date -d "$backfill_date" >/dev/null 2>&1; then
    echo "invalid BACKFILL_DATE: $backfill_date; expected YYYY-MM-DD" >&2
    exit 2
  fi
fi
if [[ "$update_sources" != "1" && ( -n "$backfill_date" || -n "$infiniops_ref" || -n "$infinirt_ref" ) ]]; then
  echo "UPDATE_SOURCES=0 cannot be combined with BACKFILL_DATE, INFINIOPS_REF, or INFINIRT_REF" >&2
  exit 2
fi

host_tools_root="${BENCHMARK_TOOLS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
output_host="$output_root/$device_id/$run_date"
container_tools_root="${CONTAINER_BENCHMARK_ROOT:-/data-aisoft/infinibench-scripts}"
source_root="${INFINIOPS_SOURCE_ROOT:-/workspace/InfiniOps}"
infinirt_source_root="${INFINIRT_SOURCE_ROOT:-/workspace/InfiniRT}"
infiniops_repo_url="${INFINIOPS_REPO_URL:-https://github.com/InfiniTensor/InfiniOps.git}"
infinirt_repo_url="${INFINIRT_REPO_URL:-https://github.com/InfiniTensor/InfiniRT.git}"
infiniops_branch="${INFINIOPS_BRANCH:-master}"
infinirt_branch="${INFINIRT_BRANCH:-master}"
build_dir="${INFINIOPS_BUILD_DIR:-/workspace/.cache/infiniops-build}"
cmake_build_parallel_level="${CMAKE_BUILD_PARALLEL_LEVEL:-64}"
output_container_root="${CONTAINER_OUTPUT_ROOT:-$output_root}"
output_container="$output_container_root/$device_id/$run_date"
runner_lock="$output_root/$device_id/.infinibench-runner.lock"
runner_lock_container="$output_container_root/$device_id/.infinibench-runner.lock"
harness="$container_tools_root/scripts/full_operator_benchmark.py"
exporter="$container_tools_root/scripts/export_full_operator_report.py"
launcher_log="$output_host/benchmark-launcher.log"
monitor_log="$output_host/benchmark-monitor.log"
foreign_flag="$output_host/foreign-process.flag"
done_file="$output_host/benchmark.done"

target_ops=abs,add,addmm,amax,amin,argmax,bitwise_and,bitwise_not,bitwise_or,bmm,cat,ceil,clamp,cos,cumsum,div,eq,exp,floor,gather,ge,gelu,gt,hardtanh,index_select,le,log,log_softmax,lt,matmul,maximum,mean,minimum,mm,mul,ne,neg,pow,reciprocal,relu,round,rsqrt,sigmoid,sign,silu,sin,softmax,sort,sqrt,sub,sum,tanh,topk
small_ops=abs,add,avg_pool2d,bmm,gather,fft_fft,gelu,matmul,mm,mul,relu,sigmoid,silu,softmax,special_gammaln,sparse_sampled_addmm,sum,where
decode_4k_ops="${METAX_DECODE_4K_OPS:-all}"
decode_4k_small_ops="${METAX_DECODE_4K_SMALL_OPS:-$small_ops}"
decode_8k_ops="${METAX_DECODE_8K_OPS:-$target_ops}"
prefill_4k_ops="${METAX_PREFILL_4K_OPS:-$target_ops}"

if ! docker inspect "$container" >/dev/null 2>&1; then
  echo "container not found: $container" >&2
  exit 2
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" != "true" ]]; then
  echo "container is not running: $container" >&2
  exit 2
fi
container_id="$(docker inspect -f '{{.Id}}' "$container")"

# Backfill changes shared source trees, so serialize all runner executions on
# this machine. The lock is released automatically when this process exits.
docker exec "$container" mkdir -p "$(dirname "$runner_lock_container")"
docker exec "$container" touch "$runner_lock_container"
docker exec "$container" chmod 0666 "$runner_lock_container"
exec 9>"$runner_lock"
if ! flock -n 9; then
  echo "another MetaX benchmark runner is active: $runner_lock" >&2
  exit 2
fi

# Create the run directory from the container so a shared frontend/agent
# account does not need write permission on the parent NFS directory. The
# launcher writes logs from the host after this directory is prepared.
docker exec "$container" mkdir -p "$output_container"
docker exec "$container" chmod 01777 "$output_container"
mkdir -p "$output_host"
: >"$launcher_log"
: >"$monitor_log"
: >"$foreign_flag"
: >"$done_file"

docker exec "$container" test -f "$harness"
docker exec "$container" test -f "$exporter"

# Verify that the host and container output roots resolve to the same bind
# mount. The container owns the run directory on NFS, so create and remove the
# marker from the container. A host-side unlink can fail with EPERM in a sticky
# directory when the runner is invoked by another shared account.
output_path_marker=".benchmark-output-path-check-$$-$(date +%s%N)"
docker exec "$container" touch "$output_container/$output_path_marker"
if ! test -f "$output_host/$output_path_marker"; then
  docker exec "$container" rm -f -- "$output_container/$output_path_marker"
  echo "OUTPUT_ROOT and CONTAINER_OUTPUT_ROOT do not point to the same mounted directory" >&2
  exit 2
fi
docker exec "$container" rm -f -- "$output_container/$output_path_marker"

prepare_source_repo() {
  local label="$1"
  local path="$2"
  local url="$3"
  local branch="$4"
  local requested_ref="$5"
  local source_date="$6"
  local cutoff requested_commit status_output resolved_ref

  echo "=== source $label prepare start=$(date --iso-8601=seconds) ===" | tee -a "$launcher_log"
  echo "repo=$url branch=$branch path=$path" | tee -a "$launcher_log"

  if ! docker exec "$container" test -d "$path/.git"; then
    if docker exec "$container" test -e "$path"; then
      echo "$label source path exists but is not a Git repository: $path" | tee -a "$launcher_log" >&2
      return 2
    fi
    echo "action=clone" | tee -a "$launcher_log"
    if ! docker exec "$container" mkdir -p "$(dirname "$path")"; then
      return 2
    fi
    if ! docker exec "$container" git clone --branch "$branch" "$url" "$path" 2>&1 | tee -a "$launcher_log"; then
      echo "$label clone failed" | tee -a "$launcher_log" >&2
      return 2
    fi
  else
    echo "action=fetch" | tee -a "$launcher_log"
    if ! status_output="$(docker exec "$container" git -c "safe.directory=$path" -C "$path" status --porcelain=v1)"; then
      printf '%s\n' "$status_output" | tee -a "$launcher_log" >&2
      return 2
    fi
    if [[ -n "$status_output" ]]; then
      echo "$label repository has local changes; refusing to overwrite:" | tee -a "$launcher_log" >&2
      printf '%s\n' "$status_output" | tee -a "$launcher_log" >&2
      return 2
    fi
    if ! docker exec "$container" git -c "safe.directory=$path" -C "$path" remote set-url origin "$url" 2>&1 | tee -a "$launcher_log"; then
      return 2
    fi
    if ! docker exec "$container" git -c "safe.directory=$path" -C "$path" fetch --prune --tags origin 2>&1 | tee -a "$launcher_log"; then
      return 2
    fi
  fi

  if ! docker exec "$container" git -c "safe.directory=$path" -C "$path" show-ref --verify --quiet "refs/remotes/origin/$branch"; then
    echo "$label remote branch not found: origin/$branch" | tee -a "$launcher_log" >&2
    return 2
  fi

  if [[ -n "$requested_ref" ]]; then
    echo "version=explicit ref=$requested_ref" | tee -a "$launcher_log"
    if ! resolved_ref="$(docker exec "$container" git -c "safe.directory=$path" -C "$path" rev-parse --verify "$requested_ref^{commit}")"; then
      echo "$label ref was not found after fetch: $requested_ref" | tee -a "$launcher_log" >&2
      return 2
    fi
    if ! docker exec "$container" git -c "safe.directory=$path" -C "$path" checkout -q "$resolved_ref" 2>&1 | tee -a "$launcher_log"; then
      return 2
    fi
  elif [[ -n "$source_date" ]]; then
    cutoff="${source_date}T23:59:59+08:00"
    echo "version=backfill date=$source_date cutoff=$cutoff" | tee -a "$launcher_log"
    if ! requested_commit="$(docker exec "$container" git -c "safe.directory=$path" -C "$path" rev-list -1 --before="$cutoff" "refs/remotes/origin/$branch")"; then
      echo "$label failed to resolve a commit before $cutoff" | tee -a "$launcher_log" >&2
      return 2
    fi
    if [[ -z "$requested_commit" ]]; then
      echo "no $label commit exists before $cutoff on origin/$branch" | tee -a "$launcher_log" >&2
      return 2
    fi
    echo "$label selected_commit=$requested_commit" | tee -a "$launcher_log"
    if ! docker exec "$container" git -c "safe.directory=$path" -C "$path" checkout -q "$requested_commit" 2>&1 | tee -a "$launcher_log"; then
      return 2
    fi
  else
    echo "version=latest" | tee -a "$launcher_log"
    if docker exec "$container" git -c "safe.directory=$path" -C "$path" show-ref --verify --quiet "refs/heads/$branch"; then
      if ! docker exec "$container" git -c "safe.directory=$path" -C "$path" checkout -q "$branch" 2>&1 | tee -a "$launcher_log"; then
        return 2
      fi
      if ! docker exec "$container" git -c "safe.directory=$path" -C "$path" pull --ff-only origin "$branch" 2>&1 | tee -a "$launcher_log"; then
        return 2
      fi
    else
      if ! docker exec "$container" git -c "safe.directory=$path" -C "$path" checkout -q -b "$branch" "refs/remotes/origin/$branch" 2>&1 | tee -a "$launcher_log"; then
        return 2
      fi
    fi
  fi

  echo "$label commit=$(docker exec "$container" git -c "safe.directory=$path" -C "$path" rev-parse HEAD)" | tee -a "$launcher_log"
  echo "=== source $label prepare end=$(date --iso-8601=seconds) ===" | tee -a "$launcher_log"
}

if [[ "$update_sources" == "1" ]]; then
  prepare_source_repo "InfiniOps" "$source_root" "$infiniops_repo_url" "$infiniops_branch" "$infiniops_ref" "$backfill_date"
  prepare_source_repo "InfiniRT" "$infinirt_source_root" "$infinirt_repo_url" "$infinirt_branch" "$infinirt_ref" "$backfill_date"
else
  {
    echo "source_update=disabled"
    docker exec "$container" test -d "$source_root/.git"
    docker exec "$container" test -d "$infinirt_source_root/.git"
  } | tee -a "$launcher_log"
fi

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
  echo "execution_time=$execution_time"
  echo "run_date=$run_date"
  echo "csv_date=$csv_date"
  echo "source_update=$update_sources"
  echo "source_backfill_date=$backfill_date"
  echo "infiniops_repo=$infiniops_repo_url"
  echo "infiniops_branch=$infiniops_branch"
  echo "infiniops_requested_ref=$infiniops_ref"
  echo "infinirt_repo=$infinirt_repo_url"
  echo "infinirt_branch=$infinirt_branch"
  echo "infinirt_requested_ref=$infinirt_ref"
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
  echo "execution_time=$execution_time"
  echo "csv_date=$csv_date"
  echo "source_update=$update_sources"
  echo "source_backfill_date=$backfill_date"
  echo "infiniops_requested_ref=$infiniops_ref"
  echo "infinirt_requested_ref=$infinirt_ref"
  echo "sha=$sha"
  echo "infinirt_sha=$infinirt_sha"
} >"$done_file"

wait "$monitor_pid"
if [[ "$foreign_process_seen" != 0 ]]; then
  exit 3
fi
exit "$exit_code"
