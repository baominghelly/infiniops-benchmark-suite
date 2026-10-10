#!/usr/bin/env bash
set -euo pipefail

container="${CONTAINER:-baoming_infiniops_hygon}"
requested_gpu="${1:-${PHYSICAL_GPU:-}}"
auto_select_gpu="${AUTO_SELECT_GPU:-1}"
physical_gpu=""
gpu_selection=""
gpu_candidates=""
gpu_busy=""
gpu_rejected=""
gpu_max_utilization="${GPU_MAX_UTILIZATION_PERCENT:-5}"
gpu_min_free_memory="${GPU_MIN_FREE_MEMORY_PERCENT:-0}"
gpu_allow_occupied="${GPU_ALLOW_OCCUPIED:-0}"

execution_time="${EXECUTION_TIME:-$(date +%Y%m%d-%H%M%S)}"
execution_date="${execution_time:0:8}"
tester="${TESTER:-baoming}"
build_infiniops="${BUILD_INFINIOPS:-1}"
hygon_arch="${HYGON_ARCH:-gfx906}"
hygon_cuda_root="${HYGON_CUDA_ROOT:-/opt/dtk/cuda/cuda-12}"
hygon_dtk_root="${HYGON_DTK_ROOT:-/workspace/.cache/dtk-$(basename "$hygon_cuda_root")}"
runtime_ld_library_path="$hygon_cuda_root/lib64:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/dtk/rccl/lib:/opt/ucx/lib:/opt/mpi/lib:/opt/hwloc/lib"
pybind11_version="${PYBIND11_VERSION:-3.1.0}"
worker_timeout="${WORKER_TIMEOUT:-60}"
worker_batch_size="${WORKER_BATCH_SIZE:-8}"
worker_batch_timeout="${WORKER_BATCH_TIMEOUT:-120}"
update_sources="${UPDATE_SOURCES:-1}"
backfill_date="${BACKFILL_DATE:-}"
reuse_run_dir="${REUSE_RUN_DIR:-0}"
replace_result_dir="${REPLACE_RESULT_DIR:-0}"
reset_stale_build_dir="${RESET_STALE_BUILD_DIR:-1}"
clear_build_proxy="${CLEAR_BUILD_PROXY:-1}"
source_preflight="${SOURCE_PREFLIGHT:-1}"
build_http_proxy="${BUILD_HTTP_PROXY:-}"
build_https_proxy="${BUILD_HTTPS_PROXY:-}"
build_all_proxy="${BUILD_ALL_PROXY:-}"
build_no_proxy="${BUILD_NO_PROXY:-}"
infiniops_ref="${INFINIOPS_REF:-}"
infinirt_ref="${INFINIRT_REF:-}"
device_id="${DEVICE_ID:-hygon-bw1000}"
output_csv_name="${OUTPUT_CSV_NAME:-infiniops-$device_id.csv}"
output_root="${OUTPUT_ROOT:-/data-aisoft/infinibench_data/ops}"
min_run_time="${MIN_RUN_TIME:-0.02}"
benchmark_iterations="${BENCHMARK_ITERATIONS:-0}"
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
if [[ ! "$min_run_time" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v value="$min_run_time" 'BEGIN { exit !(value > 0) }'; then
  echo "invalid MIN_RUN_TIME: $min_run_time; expected a positive number of seconds" >&2
  exit 2
fi
if [[ ! "$benchmark_iterations" =~ ^[0-9]+$ ]]; then
  echo "invalid BENCHMARK_ITERATIONS: $benchmark_iterations; expected a non-negative integer" >&2
  exit 2
fi
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

case "$gpu_allow_occupied" in
  0|1) ;;
  *)
    echo "invalid GPU_ALLOW_OCCUPIED: $gpu_allow_occupied; expected 0 or 1" >&2
    exit 2
    ;;
esac
for gpu_threshold in "$gpu_max_utilization" "$gpu_min_free_memory"; do
  if [[ ! "$gpu_threshold" =~ ^[0-9]+$ ]] || (( gpu_threshold > 100 )); then
    echo "invalid GPU threshold: $gpu_threshold; expected an integer between 0 and 100" >&2
    exit 2
  fi
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
  run_date="backfill-${backfill_date//-/}-${execution_date}"
  csv_date="${backfill_date//-//}"
else
  run_date="$execution_date"
  csv_date="${execution_time:0:4}/${execution_time:4:2}/${execution_time:6:2}"
fi

case "$update_sources" in
  0|1) ;;
  *)
    echo "invalid UPDATE_SOURCES: $update_sources; expected 0 or 1" >&2
    exit 2
    ;;
esac
case "$reuse_run_dir" in
  0|1) ;;
  *)
    echo "invalid REUSE_RUN_DIR: $reuse_run_dir; expected 0 or 1" >&2
    exit 2
    ;;
esac
case "$replace_result_dir" in
  0|1) ;;
  *)
    echo "invalid REPLACE_RESULT_DIR: $replace_result_dir; expected 0 or 1" >&2
    exit 2
    ;;
esac
if [[ "$replace_result_dir" == "1" && "$reuse_run_dir" != "1" ]]; then
  echo "REPLACE_RESULT_DIR=1 requires REUSE_RUN_DIR=1" >&2
  exit 2
fi
for build_gate in "$reset_stale_build_dir" "$clear_build_proxy" "$source_preflight"; do
  case "$build_gate" in
    0|1) ;;
    *)
      echo "invalid build gate: $build_gate; expected RESET_STALE_BUILD_DIR, CLEAR_BUILD_PROXY, or SOURCE_PREFLIGHT to be 0 or 1" >&2
      exit 2
      ;;
  esac
done
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

host_tools_root="${BENCHMARK_TOOLS_ROOT:-/data-aisoft/infinibench-scripts}"
output_host="$output_root/$device_id/$run_date"
container_tools_root="${CONTAINER_BENCHMARK_ROOT:-/data-aisoft/infinibench-scripts}"
source_root="${INFINIOPS_SOURCE_ROOT:-/workspace/InfiniOps}"
infinirt_source_root="${INFINIRT_SOURCE_ROOT:-/workspace/InfiniRT}"
infiniops_repo_urls_input="${INFINIOPS_REPO_URLS:-${INFINIOPS_REPO_URL:-https://github.com/InfiniTensor/InfiniOps.git,https://gh-proxy.com/https://github.com/InfiniTensor/InfiniOps.git,https://ghproxy.net/https://github.com/InfiniTensor/InfiniOps.git}}"
infinirt_repo_urls_input="${INFINIRT_REPO_URLS:-${INFINIRT_REPO_URL:-https://github.com/InfiniTensor/InfiniRT.git,https://gh-proxy.com/https://github.com/InfiniTensor/InfiniRT.git,https://ghproxy.net/https://github.com/InfiniTensor/InfiniRT.git}}"
source_probe_timeout="${SOURCE_PROBE_TIMEOUT_SECONDS:-30}"
source_operation_timeout="${SOURCE_OPERATION_TIMEOUT_SECONDS:-900}"
infiniops_branch="${INFINIOPS_BRANCH:-master}"
infinirt_branch="${INFINIRT_BRANCH:-master}"
for timeout_value in "$source_probe_timeout" "$source_operation_timeout"; do
  if [[ ! "$timeout_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid source timeout: $timeout_value; expected a positive integer" >&2
    exit 2
  fi
done
build_dir="${INFINIOPS_BUILD_DIR:-/workspace/.cache/infiniops-hygon-build}"
infinirt_build_dir="${INFINIRT_BUILD_DIR:-/workspace/.cache/infinirt-hygon-build}"
infinirt_install_root="${INFINIRT_INSTALL_ROOT:-/workspace/.cache/infinirt-hygon-install}"
cmake_build_parallel_level="${CMAKE_BUILD_PARALLEL_LEVEL:-64}"
infiniops_selected_url="not-updated"
infinirt_selected_url="not-updated"
pybind11_version_detected=""
proxy_sanitized="$([[ "$clear_build_proxy" == "1" ]] && echo 1 || echo 0)"
build_cache_reset=""
network_env_prefix=(env)
if [[ "$clear_build_proxy" == "1" ]]; then
  for proxy_var in HTTP_PROXY HTTPS_PROXY ALL_PROXY FTP_PROXY NO_PROXY http_proxy https_proxy all_proxy ftp_proxy no_proxy; do
    case "$proxy_var" in
      HTTP_PROXY|http_proxy)
        if [[ -n "$build_http_proxy" ]]; then continue; fi
        ;;
      HTTPS_PROXY|https_proxy)
        if [[ -n "$build_https_proxy" ]]; then continue; fi
        ;;
      ALL_PROXY|all_proxy)
        if [[ -n "$build_all_proxy" ]]; then continue; fi
        ;;
      NO_PROXY|no_proxy)
        if [[ -n "$build_no_proxy" ]]; then continue; fi
        ;;
    esac
    network_env_prefix+=(-u "$proxy_var")
  done
fi
build_proxy_args=()
if [[ -n "$build_http_proxy" ]]; then
  build_proxy_args+=(-e HTTP_PROXY="$build_http_proxy" -e http_proxy="$build_http_proxy")
fi
if [[ -n "$build_https_proxy" ]]; then
  build_proxy_args+=(-e HTTPS_PROXY="$build_https_proxy" -e https_proxy="$build_https_proxy")
fi
if [[ -n "$build_all_proxy" ]]; then
  build_proxy_args+=(-e ALL_PROXY="$build_all_proxy" -e all_proxy="$build_all_proxy")
fi
if [[ -n "$build_no_proxy" ]]; then
  build_proxy_args+=(-e NO_PROXY="$build_no_proxy" -e no_proxy="$build_no_proxy")
fi
output_container_root="${CONTAINER_OUTPUT_ROOT:-$output_root}"
output_container="$output_container_root/$device_id/$run_date"
runner_lock="$output_root/$device_id/.infinibench-runner.lock"
runner_lock_container="$output_container_root/$device_id/.infinibench-runner.lock"
harness="$container_tools_root/scripts/full_operator_benchmark.py"
exporter="$container_tools_root/scripts/export_full_operator_report.py"
partial_exporter="$container_tools_root/scripts/export_partial_results.py"
gpu_audit="$host_tools_root/scripts/hygon_gpu_audit.py"
launcher_log="$output_host/benchmark-launcher.log"
monitor_log="$output_host/benchmark-monitor.log"
foreign_flag="$output_host/foreign-process.flag"
done_file="$output_host/benchmark.done"
stage="preflight"
exit_code=0

target_ops=abs,add,addmm,amax,amin,argmax,bitwise_and,bitwise_not,bitwise_or,bmm,cat,ceil,clamp,cos,cumsum,div,eq,exp,floor,gather,ge,gelu,gt,hardtanh,index_select,le,log,log_softmax,lt,matmul,maximum,mean,minimum,mm,mul,ne,neg,pow,reciprocal,relu,round,rsqrt,sigmoid,sign,silu,sin,softmax,sort,sqrt,sub,sum,tanh,topk
small_ops=abs,add,avg_pool2d,bmm,gather,fft_fft,gelu,matmul,mm,mul,relu,sigmoid,silu,softmax,special_gammaln,sparse_sampled_addmm,sum,where
decode_4k_ops="${HYGON_DECODE_4K_OPS:-all}"
decode_4k_small_ops="${HYGON_DECODE_4K_SMALL_OPS:-$small_ops}"
decode_8k_ops="${HYGON_DECODE_8K_OPS:-$target_ops}"
prefill_4k_ops="${HYGON_PREFILL_4K_OPS:-$target_ops}"

"$host_tools_root/scripts/ensure_benchmark_container.sh" "$container"
container_id="$(docker inspect -f '{{.Id}}' "$container")"

docker_exec_network() {
  docker exec "${build_proxy_args[@]}" "$container" "${network_env_prefix[@]}" "$@"
}

probe_source_urls() {
  local label="$1"
  local urls_input="$2"
  local -a urls
  local url reachable=0

  read -r -a urls <<<"${urls_input//,/ }"
  if [[ "${#urls[@]}" -eq 0 ]]; then
    echo "$label repository URL list is empty" >&2
    return 2
  fi
  for url in "${urls[@]}"; do
    echo "$label source_preflight_candidate=$url" >&2
    if docker_exec_network timeout "$source_probe_timeout" git ls-remote "$url" HEAD >/dev/null 2>&1; then
      echo "$label source_preflight_reachable=$url" >&2
      reachable=1
      break
    fi
  done
  if [[ "$reachable" != "1" ]]; then
    echo "$label source preflight failed: no configured repository URL is reachable" >&2
    return 2
  fi
}

validate_hygon_cuda_root() {
  local path
  for path in "$hygon_cuda_root/bin/nvcc" "$hygon_cuda_root/include" "$hygon_cuda_root/lib64"; do
    if ! docker exec "$container" test -e "$path"; then
      echo "invalid HYGON_CUDA_ROOT=$hygon_cuda_root: missing $path" >&2
      return 2
    fi
  done
  if ! docker exec "$container" test -x "$hygon_cuda_root/bin/nvcc"; then
    echo "invalid HYGON_CUDA_ROOT=$hygon_cuda_root: bin/nvcc is not executable" >&2
    return 2
  fi
}

if [[ "$build_infiniops" == "1" ]]; then
  if ! validate_hygon_cuda_root; then
    exit 2
  fi
fi
if [[ "$source_preflight" == "1" ]]; then
  if ! probe_source_urls "InfiniOps" "$infiniops_repo_urls_input"; then
    exit 2
  fi
  if ! probe_source_urls "InfiniRT" "$infinirt_repo_urls_input"; then
    exit 2
  fi
fi

# Backfill changes shared source trees, so serialize all runner executions on
# this machine. The lock is released automatically when this process exits.
docker exec "$container" mkdir -p "$(dirname "$runner_lock_container")"
docker exec "$container" touch "$runner_lock_container"
docker exec "$container" chmod 0666 "$runner_lock_container"
exec 9>"$runner_lock"
if ! flock -n 9; then
  echo "another Hygon/DCU benchmark runner is active: $runner_lock" >&2
  exit 2
fi

# A normal run directory is deterministic for one execution date, and a
# backfill directory is deterministic for one target/execution date pair.
# Refuse to mix a new attempt with stale checkpoints. REUSE_RUN_DIR=1
# explicitly replaces only this fixed directory for a retry.
if [[ -e "$output_host" ]] && [[ "$reuse_run_dir" != "1" ]]; then
  if [[ -n "$(find "$output_host" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "output directory already exists and is non-empty: $output_host" >&2
    echo "set REUSE_RUN_DIR=1 to replace it for the same run date" >&2
    exit 2
  fi
fi
if [[ "$reuse_run_dir" == "1" && -e "$output_container" ]]; then
  case "$output_container" in
    /data-aisoft/infinibench_data/ops/*/*) ;;
    *)
      echo "refusing to reset untrusted output path: $output_container" >&2
      exit 2
      ;;
  esac
  if [[ "$replace_result_dir" != "1" ]] && docker exec "$container" python3 -c '
import pathlib, sys
root = pathlib.Path(sys.argv[1])
found = any(path.name in {"results.json", "results.jsonl"} for path in root.rglob("*") if path.is_file())
raise SystemExit(0 if found else 1)
' "$output_container"; then
    echo "output directory contains complete or checkpoint result files: $output_container" >&2
    echo "set REUSE_RUN_DIR=1 REPLACE_RESULT_DIR=1 to replace it explicitly" >&2
    exit 2
  fi
  docker exec "$container" rm -rf -- "$output_container"
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

finish() {
  local rc=$?
  if [[ ! -s "$done_file" ]]; then
    {
      echo "exit_code=$rc"
      echo "stage=$stage"
      echo "execution_time=$execution_time"
      echo "execution_date=$execution_date"
      echo "run_date=$run_date"
      echo "csv_date=$csv_date"
      echo "source_backfill_date=$backfill_date"
      echo "selected_source_urls=$infiniops_selected_url,$infinirt_selected_url"
      echo "proxy_sanitized=$proxy_sanitized"
      echo "pybind11_version=$pybind11_version_detected"
      echo "hygon_cuda_root=$hygon_cuda_root"
      echo "build_cache_reset=$build_cache_reset"
      echo "gpu_max_utilization_percent=$gpu_max_utilization"
      echo "gpu_min_free_memory_percent=$gpu_min_free_memory"
      echo "gpu_allow_occupied=$gpu_allow_occupied"
      echo "foreign_process_seen=$([[ -s "$foreign_flag" ]] && echo 1 || echo 0)"
      echo "end=$(date --iso-8601=seconds)"
    } >"$done_file"
  fi
}
trap finish EXIT

docker exec "$container" test -f "$harness"
docker exec "$container" test -f "$exporter"
docker exec "$container" test -f "$partial_exporter"

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
  local urls_input="$3"
  local branch="$4"
  local requested_ref="$5"
  local source_date="$6"
  local cutoff requested_commit status_output resolved_ref selected_url=""
  local -a urls
  local url clone_target

  read -r -a urls <<<"${urls_input//,/ }"
  if [[ "${#urls[@]}" -eq 0 ]]; then
    echo "$label repository URL list is empty" | tee -a "$launcher_log" >&2
    return 2
  fi

  echo "=== source $label prepare start=$(date --iso-8601=seconds) ===" | tee -a "$launcher_log"
  echo "repo_urls=$urls_input branch=$branch path=$path" | tee -a "$launcher_log"

  if ! docker exec "$container" test -d "$path/.git"; then
    if docker exec "$container" test -e "$path"; then
      echo "$label source path exists but is not a Git repository: $path" | tee -a "$launcher_log" >&2
      return 2
    fi
    echo "action=clone" | tee -a "$launcher_log"
    if ! docker exec "$container" mkdir -p "$(dirname "$path")"; then
      return 2
    fi
    clone_target="${path}.clone-$$"
    docker exec "$container" test ! -e "$clone_target"
    for url in "${urls[@]}"; do
      echo "clone_candidate=$url" | tee -a "$launcher_log"
      if ! docker_exec_network timeout "$source_probe_timeout" git ls-remote "$url" HEAD >/dev/null 2>&1; then
        echo "clone_candidate_unreachable=$url" | tee -a "$launcher_log"
        continue
      fi
      if docker_exec_network timeout "$source_operation_timeout" git clone --branch "$branch" "$url" "$clone_target" 2>&1 | tee -a "$launcher_log"; then
        docker exec "$container" mv "$clone_target" "$path"
        selected_url="$url"
        break
      fi
      echo "clone_candidate_failed=$url" | tee -a "$launcher_log" >&2
      docker exec "$container" rm -rf -- "$clone_target"
    done
    if [[ -z "$selected_url" ]]; then
      echo "$label clone failed for all repository URLs" | tee -a "$launcher_log" >&2
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
    for url in "${urls[@]}"; do
      echo "fetch_candidate=$url" | tee -a "$launcher_log"
      if ! docker_exec_network timeout "$source_probe_timeout" git ls-remote "$url" HEAD >/dev/null 2>&1; then
        echo "fetch_candidate_unreachable=$url" | tee -a "$launcher_log"
        continue
      fi
      if ! docker_exec_network git -c "safe.directory=$path" -C "$path" remote set-url origin "$url" 2>&1 | tee -a "$launcher_log"; then
        continue
      fi
      if docker_exec_network timeout "$source_operation_timeout" git -c "safe.directory=$path" -C "$path" fetch --prune --tags origin 2>&1 | tee -a "$launcher_log"; then
        selected_url="$url"
        break
      fi
      echo "fetch_candidate_failed=$url" | tee -a "$launcher_log" >&2
    done
    if [[ -z "$selected_url" ]]; then
      echo "$label fetch failed for all repository URLs" | tee -a "$launcher_log" >&2
      return 2
    fi
  fi

  echo "$label selected_repo=$selected_url" | tee -a "$launcher_log"
  case "$label" in
    InfiniOps) infiniops_selected_url="$selected_url" ;;
    InfiniRT) infinirt_selected_url="$selected_url" ;;
  esac

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
      if ! docker exec "$container" git -c "safe.directory=$path" -C "$path" merge --ff-only "refs/remotes/origin/$branch" 2>&1 | tee -a "$launcher_log"; then
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
  prepare_source_repo "InfiniOps" "$source_root" "$infiniops_repo_urls_input" "$infiniops_branch" "$infiniops_ref" "$backfill_date"
  prepare_source_repo "InfiniRT" "$infinirt_source_root" "$infinirt_repo_urls_input" "$infinirt_branch" "$infinirt_ref" "$backfill_date"
else
  {
    echo "source_update=disabled"
    docker exec "$container" test -d "$source_root/.git"
    docker exec "$container" test -d "$infinirt_source_root/.git"
  } | tee -a "$launcher_log"
fi

select_idle_gpu() {
  local selection_output
  local -a audit_args=(
    --select
    --max-utilization "$gpu_max_utilization"
    --min-free-memory "$gpu_min_free_memory"
  )
  if [[ "$gpu_allow_occupied" == "1" ]]; then
    audit_args+=(--allow-occupied)
  fi
  if ! selection_output="$(
    python3 "$gpu_audit" "${audit_args[@]}" 2>&1
  )"; then
    printf '%s\n' "$selection_output" >&2
    return 2
  fi
  mapfile -t selection_rows <<<"$selection_output"
  if [[ "${#selection_rows[@]}" -ne 4 ]]; then
    echo "unexpected idle-HCU selector output: $selection_output" >&2
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
  echo "gpu_max_utilization_percent=$gpu_max_utilization"
  echo "gpu_min_free_memory_percent=$gpu_min_free_memory"
  echo "gpu_allow_occupied=$gpu_allow_occupied"
} >>"$launcher_log"

if [[ "$build_infiniops" == "1" ]]; then
  prepare_hygon_dtk_root() {
    local real_dtk_root="/opt/dtk"

    case "$hygon_dtk_root" in
      /workspace/.cache/*) ;;
      *)
        echo "refusing to prepare untrusted Hygon DTK staging root: $hygon_dtk_root" >&2
        return 2
        ;;
    esac
    if ! docker exec "$container" mkdir -p "$hygon_dtk_root"; then
      return 2
    fi
    if ! docker exec -i "$container" python3 - "$hygon_dtk_root" "$real_dtk_root" "$hygon_cuda_root" <<'PY'
from pathlib import Path
import sys

staging = Path(sys.argv[1]).resolve()
real_dtk = Path(sys.argv[2]).resolve()
cuda_root = Path(sys.argv[3]).resolve()
expected_parent = Path("/workspace/.cache").resolve()
if staging.parent != expected_parent:
    raise SystemExit(f"untrusted DTK staging root: {staging}")
if not real_dtk.is_dir() or not (cuda_root / "bin" / "nvcc").is_file():
    raise SystemExit("real DTK root or Hygon CUDA root is invalid")
staging.mkdir(parents=True, exist_ok=True)
if staging.exists() and not staging.is_dir():
    raise SystemExit(f"DTK staging path is not a directory: {staging}")
for source in real_dtk.iterdir():
    if source.name == "cuda":
        continue
    target = staging / source.name
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        target.unlink()
    if target.is_dir() and not target.is_symlink():
        raise SystemExit(f"unexpected real directory in DTK staging root: {target}")
    target.symlink_to(source)
cuda_link = staging / "cuda"
if cuda_link.is_symlink() or (cuda_link.exists() and not cuda_link.is_dir()):
    cuda_link.unlink()
if cuda_link.is_dir() and not cuda_link.is_symlink():
    raise SystemExit(f"unexpected cuda directory in DTK staging root: {cuda_link}")
cuda_link.symlink_to(cuda_root)
PY
    then
      echo "failed to prepare deterministic Hygon DTK staging root: $hygon_dtk_root" >&2
      return 2
    fi
    docker exec "$container" test -x "$hygon_dtk_root/cuda/bin/nvcc"
  }

  cmake_cache_value() {
    local cache_file="$1"
    local key="$2"
    docker exec "$container" sh -c '
      grep -m1 "^$2:" "$1" | cut -d= -f2-
    ' sh "$cache_file" "$key"
  }

  normalize_container_path() {
    docker exec "$container" readlink -m -- "$1"
  }

  check_build_cache() {
    local mode="$1"
    local label="$2"
    local build_cache_dir="$3"
    shift 3
    local -a expected_settings=("$@")
    local setting key expected actual expected_path actual_path
    local -a mismatches=()

    if ! docker exec "$container" test -e "$build_cache_dir"; then
      return 0
    fi
    if ! docker exec "$container" test -f "$build_cache_dir/CMakeCache.txt"; then
      mismatches+=("CMakeCache.txt=missing")
    else
      for setting in "${expected_settings[@]}"; do
        key="${setting%%=*}"
        expected="${setting#*=}"
        actual="$(cmake_cache_value "$build_cache_dir/CMakeCache.txt" "$key")"
        if [[ -z "$actual" ]]; then
          mismatches+=("$key=missing (expected $expected)")
          continue
        fi
        expected_path="$(normalize_container_path "$expected")"
        actual_path="$(normalize_container_path "$actual")"
        if [[ "$actual_path" != "$expected_path" ]]; then
          mismatches+=("$key=$actual_path (expected $expected_path)")
        fi
      done
    fi

    if [[ "${#mismatches[@]}" -eq 0 ]]; then
      return 0
    fi

    printf 'stale %s CMake cache: %s\n' "$label" "${mismatches[*]}" | tee -a "$launcher_log" >&2
    if [[ "$mode" == "validate" || "$reset_stale_build_dir" != "1" ]]; then
      if [[ "$reset_stale_build_dir" != "1" ]]; then
        echo "set RESET_STALE_BUILD_DIR=1 to reset this exact build directory, or fix the cache manually" | tee -a "$launcher_log" >&2
      fi
      return 2
    fi

    if docker exec "$container" test -L "$build_cache_dir"; then
      echo "refusing to reset a symlinked build directory: $build_cache_dir" | tee -a "$launcher_log" >&2
      return 2
    fi
    local resolved_build_cache_dir
    resolved_build_cache_dir="$(normalize_container_path "$build_cache_dir")"
    case "$resolved_build_cache_dir" in
      /workspace/.cache/*) ;;
      *)
        echo "refusing to reset untrusted build directory: $resolved_build_cache_dir" | tee -a "$launcher_log" >&2
        return 2
        ;;
    esac
    docker exec "$container" rm -rf -- "$resolved_build_cache_dir"
    if [[ -n "$build_cache_reset" ]]; then
      build_cache_reset="$build_cache_reset,$label"
    else
      build_cache_reset="$label"
    fi
    echo "$label stale_build_cache_reset=1" | tee -a "$launcher_log"
  }

  stage="build-dependencies"
  if ! docker exec "${build_proxy_args[@]}" -e LD_LIBRARY_PATH="$runtime_ld_library_path" "$container" "${network_env_prefix[@]}" \
    python3 -c "import clang, packaging, scikit_build_core, yaml" >/dev/null 2>&1; then
    docker exec "${build_proxy_args[@]}" -e LD_LIBRARY_PATH="$runtime_ld_library_path" "$container" "${network_env_prefix[@]}" \
      python3 -m pip install scikit-build-core libclang packaging pyyaml "pybind11==$pybind11_version" \
      2>&1 | tee -a "$launcher_log"
  fi
  if pybind11_version_detected="$(docker exec -e LD_LIBRARY_PATH="$runtime_ld_library_path" "$container" \
    python3 -c "import pybind11; print(pybind11.__version__)")" && \
    [[ "$pybind11_version_detected" == "$pybind11_version" ]] && \
    docker exec -e LD_LIBRARY_PATH="$runtime_ld_library_path" "$container" \
      python3 -m pybind11 --cmakedir >/dev/null 2>&1; then
    echo "pybind11_version=$pybind11_version_detected" | tee -a "$launcher_log"
  else
    docker exec "${build_proxy_args[@]}" -e LD_LIBRARY_PATH="$runtime_ld_library_path" "$container" "${network_env_prefix[@]}" \
      python3 -m pip install --force-reinstall "pybind11==$pybind11_version" \
      2>&1 | tee -a "$launcher_log"
    pybind11_version_detected="$(docker exec -e LD_LIBRARY_PATH="$runtime_ld_library_path" "$container" \
      python3 -c "import pybind11; print(pybind11.__version__)")"
    docker exec -e LD_LIBRARY_PATH="$runtime_ld_library_path" "$container" \
      python3 -m pybind11 --cmakedir >/dev/null
  fi

  stage="build-toolchain"
  prepare_hygon_dtk_root
  stage="build-infinirt"
  check_build_cache guard "InfiniRT" "$infinirt_build_dir" \
    "CMAKE_CUDA_COMPILER=$hygon_cuda_root/bin/nvcc" \
    "CUDAToolkit_ROOT=$hygon_cuda_root" \
    "DTK_ROOT=$hygon_dtk_root" \
    "HYGON_ARCH=$hygon_arch" \
    "WITH_CPU=ON" \
    "WITH_HYGON=ON" \
    "CMAKE_INSTALL_PREFIX=$infinirt_install_root"
  docker exec "$container" mkdir -p "$infinirt_build_dir" "$infinirt_install_root"
  {
    echo "=== InfiniRT build start=$(date --iso-8601=seconds) ==="
    echo "infinirt_build_dir=$infinirt_build_dir"
    echo "infinirt_install_root=$infinirt_install_root"
    echo "build_dir=$build_dir"
    echo "cmake_build_parallel_level=$cmake_build_parallel_level"
    echo "hygon_cuda_root=$hygon_cuda_root"
    echo "hygon_dtk_root=$hygon_dtk_root"
    echo "pybind11_version=$pybind11_version_detected"
  } >>"$launcher_log"
  build_exit=0
  docker exec \
    -e DTK_ROOT="$hygon_dtk_root" \
    -e CUDA_HOME="$hygon_cuda_root" \
    -e CUDA_PATH="$hygon_cuda_root" \
    -e HYGON_ARCH="$hygon_arch" \
    -e LD_LIBRARY_PATH="$runtime_ld_library_path" \
    "$container" cmake \
      -S "$infinirt_source_root" \
      -B "$infinirt_build_dir" \
      -DCMAKE_INSTALL_PREFIX:PATH="$infinirt_install_root" \
      -DCMAKE_CUDA_COMPILER:FILEPATH="$hygon_cuda_root/bin/nvcc" \
      -DCUDAToolkit_ROOT:PATH="$hygon_cuda_root" \
      -DDTK_ROOT:PATH="$hygon_dtk_root" \
      -DWITH_CPU:BOOL=ON \
      -DWITH_HYGON:BOOL=ON \
      -DINFINI_RT_BUILD_TESTING:BOOL=OFF \
      2>&1 | tee -a "$launcher_log" || build_exit=$?
  if [[ "$build_exit" -ne 0 ]]; then
    echo "infinirt_configure_exit=$build_exit" | tee -a "$launcher_log"
    exit "$build_exit"
  fi
  check_build_cache validate "InfiniRT" "$infinirt_build_dir" \
    "CMAKE_CUDA_COMPILER=$hygon_cuda_root/bin/nvcc" \
    "CUDAToolkit_ROOT=$hygon_cuda_root" \
    "DTK_ROOT=$hygon_dtk_root" \
    "HYGON_ARCH=$hygon_arch" \
    "WITH_CPU=ON" \
    "WITH_HYGON=ON" \
    "CMAKE_INSTALL_PREFIX=$infinirt_install_root"

  docker exec \
    -e HYGON_ARCH="$hygon_arch" \
    -e LD_LIBRARY_PATH="$runtime_ld_library_path" \
    "$container" cmake --build "$infinirt_build_dir" --parallel "$cmake_build_parallel_level" \
    2>&1 | tee -a "$launcher_log" || build_exit=$?
  if [[ "$build_exit" -ne 0 ]]; then
    echo "infinirt_build_exit=$build_exit" | tee -a "$launcher_log"
    exit "$build_exit"
  fi
  docker exec \
    -e HYGON_ARCH="$hygon_arch" \
    -e LD_LIBRARY_PATH="$runtime_ld_library_path" \
    "$container" cmake --install "$infinirt_build_dir" \
    2>&1 | tee -a "$launcher_log" || build_exit=$?
  if [[ "$build_exit" -ne 0 ]]; then
    echo "infinirt_install_exit=$build_exit" | tee -a "$launcher_log"
    exit "$build_exit"
  fi
  echo "=== InfiniRT build end=$(date --iso-8601=seconds) ===" >>"$launcher_log"

  stage="build-infiniops"
  check_build_cache guard "InfiniOps" "$build_dir" \
    "CMAKE_CUDA_COMPILER=$hygon_cuda_root/bin/nvcc" \
    "CUDAToolkit_ROOT=$hygon_cuda_root" \
    "DTK_ROOT=$hygon_dtk_root" \
    "HYGON_ARCH=$hygon_arch" \
    "WITH_CPU=ON" \
    "WITH_HYGON=ON" \
    "INFINI_RT_ROOT=$infinirt_install_root"
  docker exec "$container" mkdir -p "$build_dir"
  echo "=== InfiniOps build start=$(date --iso-8601=seconds) ===" >>"$launcher_log"
  docker exec \
    -e DTK_ROOT="$hygon_dtk_root" \
    -e CUDA_HOME="$hygon_cuda_root" \
    -e CUDA_PATH="$hygon_cuda_root" \
    -e HYGON_ARCH="$hygon_arch" \
    -e LD_LIBRARY_PATH="$runtime_ld_library_path" \
    -e CMAKE_BUILD_PARALLEL_LEVEL="$cmake_build_parallel_level" \
    -e PIP_NO_CACHE_DIR=1 \
    "$container" python3 -m pip install \
      --no-build-isolation \
      --no-deps \
      --force-reinstall \
      "$source_root" \
      "--config-settings=build-dir=$build_dir" \
      --config-settings=cmake.define.AUTO_DETECT_DEVICES:BOOL=OFF \
      --config-settings=cmake.define.AUTO_DETECT_BACKENDS:BOOL=OFF \
      --config-settings=cmake.define.WITH_CPU:BOOL=ON \
      --config-settings=cmake.define.WITH_HYGON:BOOL=ON \
      --config-settings=cmake.define.WITH_TORCH:BOOL=ON \
      --config-settings=cmake.define.WITH_LINKED:BOOL=OFF \
      "--config-settings=cmake.define.INFINI_RT_ROOT:PATH=$infinirt_install_root" \
      "--config-settings=cmake.define.CMAKE_CUDA_COMPILER:FILEPATH=$hygon_cuda_root/bin/nvcc" \
      "--config-settings=cmake.define.CUDAToolkit_ROOT:PATH=$hygon_cuda_root" \
      "--config-settings=cmake.define.DTK_ROOT:PATH=$hygon_dtk_root" \
      2>&1 | tee -a "$launcher_log" || build_exit=$?
  if [[ "$build_exit" -ne 0 ]]; then
    echo "infiniops_build_exit=$build_exit" | tee -a "$launcher_log"
    exit "$build_exit"
  fi
  check_build_cache validate "InfiniOps" "$build_dir" \
    "CMAKE_CUDA_COMPILER=$hygon_cuda_root/bin/nvcc" \
    "CUDAToolkit_ROOT=$hygon_cuda_root" \
    "DTK_ROOT=$hygon_dtk_root" \
    "HYGON_ARCH=$hygon_arch" \
    "WITH_CPU=ON" \
    "WITH_HYGON=ON" \
    "INFINI_RT_ROOT=$infinirt_install_root"
  echo "=== InfiniOps build end=$(date --iso-8601=seconds) ===" >>"$launcher_log"
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

stage="metadata-and-preflight"
if [[ -n "${INFINIOPS_METADATA:-}" ]]; then
  metadata="$INFINIOPS_METADATA"
else
  metadata="$(docker exec \
    -e LD_LIBRARY_PATH="$runtime_ld_library_path" \
    "$container" python3 -c \
    "from pathlib import Path; import infini; print(Path(infini.__file__).resolve().parent / 'torch_ops_metadata.json')")"
fi
docker exec "$container" test -f "$metadata"
sha="$(docker exec "$container" git -c "safe.directory=$source_root" -C "$source_root" rev-parse HEAD)"
infinirt_sha="$(docker exec "$container" git -c "safe.directory=$infinirt_source_root" -C "$infinirt_source_root" rev-parse HEAD)"

gpu_preflight_args=(
  --gpu "$physical_gpu"
  --assert-free
  --max-utilization "$gpu_max_utilization"
  --min-free-memory "$gpu_min_free_memory"
)
if [[ "$gpu_allow_occupied" == "1" ]]; then
  gpu_preflight_args+=(--allow-occupied)
fi
if ! python3 "$gpu_audit" "${gpu_preflight_args[@]}" 2>&1 | tee -a "$launcher_log"; then
  echo "physical Hygon HCU $physical_gpu does not satisfy the GPU resource gate" >&2
  exit 2
fi

{
  echo "start=$(date --iso-8601=seconds)"
  echo "container=$container"
  echo "container_id=$container_id"
  echo "physical_gpu=$physical_gpu"
  echo "gpu_max_utilization_percent=$gpu_max_utilization"
  echo "gpu_min_free_memory_percent=$gpu_min_free_memory"
  echo "gpu_allow_occupied=$gpu_allow_occupied"
  echo "execution_time=$execution_time"
  echo "execution_date=$execution_date"
  echo "run_date=$run_date"
  echo "csv_date=$csv_date"
  echo "source_update=$update_sources"
  echo "source_backfill_date=$backfill_date"
  echo "selected_source_urls=$infiniops_selected_url,$infinirt_selected_url"
  echo "proxy_sanitized=$proxy_sanitized"
  echo "pybind11_version=$pybind11_version_detected"
  echo "hygon_cuda_root=$hygon_cuda_root"
  echo "hygon_dtk_root=$hygon_dtk_root"
  echo "build_cache_reset=$build_cache_reset"
  echo "infiniops_repo_urls=$infiniops_repo_urls_input"
  echo "infiniops_branch=$infiniops_branch"
  echo "infiniops_requested_ref=$infiniops_ref"
  echo "infinirt_repo_urls=$infinirt_repo_urls_input"
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
  python3 "$gpu_audit" --gpu "$physical_gpu" || true
  echo "=== container preflight ==="
  docker exec \
    -e HIP_VISIBLE_DEVICES="$physical_gpu" \
    -e HYGON_ARCH="$hygon_arch" \
    -e LD_LIBRARY_PATH="$runtime_ld_library_path" \
    "$container" python3 -c \
    "import torch, infini.ops; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print('torch', torch.__version__); print('hip', torch.version.hip); print('hcu_count', torch.cuda.device_count()); print('device', torch.cuda.get_device_name(0)); print('infini_ops', infini.ops.__file__)"
} >>"$launcher_log" 2>&1

monitor() {
  local -a audit_args=(
    --gpu "$physical_gpu" \
    --container-id "$container_id" \
    --container "$container" \
    --harness "$harness" \
    --watch \
    --done-file "$done_file" \
    --foreign-flag "$foreign_flag"
  )
  if [[ "$gpu_allow_occupied" == "1" ]]; then
    audit_args+=(--allow-occupied)
  fi
  python3 "$gpu_audit" "${audit_args[@]}"
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
    --platform "$device_id"
    --dtype "$dtype"
    --sha "$sha"
    --min-run-time "$min_run_time"
    --benchmark-iterations "$benchmark_iterations"
    --timeout "$worker_timeout"
    --worker-batch-size "$worker_batch_size"
    --batch-timeout "$worker_batch_timeout"
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
    -e HIP_VISIBLE_DEVICES="$physical_gpu" \
    -e LD_LIBRARY_PATH="$runtime_ld_library_path" \
    -e INFINIOPS_BENCHMARK_PROFILE="$profile" \
    "$container" "${command[@]}" 2>&1 | tee -a "$launcher_log"
  return "${PIPESTATUS[0]}"
}

stage="benchmark"
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
  stage="export"
  docker exec -e LD_LIBRARY_PATH="$runtime_ld_library_path" "$container" python3 "$exporter" \
    "${exporter_sources[@]}" \
    --output "$output_container/$output_csv_name" \
    --diagnostics "$output_container/diagnostics.csv" \
    --report "$output_container/report.md" \
    --platform hygon \
    --date "$csv_date" \
    --infiniops-commit "$sha" \
    --infinirt-commit "$infinirt_sha" \
    --tester "$tester" 2>&1 | tee -a "$launcher_log"
  exporter_exit=${PIPESTATUS[0]}
  if [[ "$exporter_exit" -ne 0 ]]; then
    exit_code="$exporter_exit"
  fi
else
  echo "no result JSON files were available for export" | tee -a "$launcher_log"
  exit_code=1
fi

if [[ "$exit_code" -ne 0 || -s "$foreign_flag" ]]; then
  if docker exec "$container" python3 -c '
import pathlib, sys
root = pathlib.Path(sys.argv[1])
found = any(path.name in {"results.json", "results.jsonl"} for path in root.rglob("*") if path.is_file())
raise SystemExit(0 if found else 1)
' "$output_container"; then
    if [[ -s "$foreign_flag" ]]; then
      partial_failure_category=foreign_process
      partial_failure_reason="foreign process detected during benchmark"
    else
      partial_failure_category=benchmark_failure
      partial_failure_reason="runner exit code $exit_code"
    fi
    stage="partial-export"
    echo "=== partial export start=$(date --iso-8601=seconds) ===" | tee -a "$launcher_log"
    partial_exit=0
    docker exec "$container" python3 "$partial_exporter" \
      --output-root "$output_container" \
      --output-name "$output_csv_name" \
      --platform hygon \
      --date "$csv_date" \
      --tester "$tester" \
      --infiniops-commit "$sha" \
      --infinirt-commit "$infinirt_sha" \
      --failure-category "$partial_failure_category" \
      --failure-reason "$partial_failure_reason" \
      2>&1 | tee -a "$launcher_log" || partial_exit=$?
    if [[ "$partial_exit" -eq 0 ]]; then
      echo "partial_csv=$output_container/${output_csv_name%.csv}.partial.csv" | tee -a "$launcher_log"
    else
      echo "partial_export_failed=$partial_exit" | tee -a "$launcher_log"
    fi
  else
    echo "partial_export_skipped=no complete or checkpoint result files" | tee -a "$launcher_log"
  fi
fi

{
  echo "=== post-run HCU audit ==="
  python3 "$gpu_audit" --gpu "$physical_gpu" || true
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
  echo "gpu_max_utilization_percent=$gpu_max_utilization"
  echo "gpu_min_free_memory_percent=$gpu_min_free_memory"
  echo "gpu_allow_occupied=$gpu_allow_occupied"
  echo "execution_time=$execution_time"
  echo "execution_date=$execution_date"
  echo "run_date=$run_date"
  echo "csv_date=$csv_date"
  echo "source_update=$update_sources"
  echo "source_backfill_date=$backfill_date"
  echo "selected_source_urls=$infiniops_selected_url,$infinirt_selected_url"
  echo "proxy_sanitized=$proxy_sanitized"
  echo "pybind11_version=$pybind11_version_detected"
  echo "hygon_cuda_root=$hygon_cuda_root"
  echo "hygon_dtk_root=$hygon_dtk_root"
  echo "build_cache_reset=$build_cache_reset"
  echo "stage=$stage"
  echo "infiniops_requested_ref=$infiniops_ref"
  echo "infinirt_requested_ref=$infinirt_ref"
  echo "sha=$sha"
  echo "infinirt_sha=$infinirt_sha"
  echo "hygon_arch=$hygon_arch"
} >"$done_file"

monitor_exit=0
wait "$monitor_pid" || monitor_exit=$?
trap - EXIT
if [[ "$foreign_process_seen" != 0 ]]; then
  exit 3
fi
exit "$exit_code"
