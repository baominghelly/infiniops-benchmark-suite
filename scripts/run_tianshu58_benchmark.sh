#!/usr/bin/env bash
set -uo pipefail

container=infinibench-stack-20260724-iluvatar
physical_gpu="${1:?physical GPU index is required}"
ixsmi=/usr/local/corex-host/bin/ixsmi
workspace=/home/libaoming/infinibench-builds/20260803-iluvatar-recovery
harness=/workspace/benchmark-tools/scripts/full_operator_benchmark.py
exporter=/workspace/benchmark-tools/scripts/export_full_operator_report.py
infini_root=/workspace/src/InfiniOps-47c1c496/build-iluvatar/install
infini_lib="$infini_root/infini"
python_path="$infini_root:/usr/local/corex-4.5.0.20260615/lib64/python3/dist-packages:/workspace/src/InfiniLM/python:/workspace/src/InfiniCore/python"
ld_library_path="$infini_lib:/opt/infini/lib:/opt/infini-rt/lib:/usr/local/corex-host/lib64:/usr/local/corex/lib64:/usr/local/corex/lib64/python3/dist-packages/torch/lib:/usr/local/openmpi/lib:/usr/local/lib"
metadata="$infini_root/infini/torch_ops_metadata.json"
run_date="$(date +%Y/%m/%d)"
run_id="$(date +%F)/run-$(date +%H%M%S)"
base_host="${RESULTS_ROOT:-$workspace/benchmark-results}/$run_id"
base="${RESULTS_CONTAINER_ROOT:-/workspace/benchmark-results}/$run_id"
launcher_log="$base_host/benchmark-launcher.log"
monitor_log="$base_host/benchmark-monitor.log"
foreign_flag="$base_host/foreign-process.flag"
done_file="$base_host/benchmark.done"
container_id="$(docker inspect --format '{{.Id}}' "$container")"
target_ops=abs,add,addmm,amax,amin,argmax,bitwise_and,bitwise_not,bitwise_or,bmm,cat,ceil,clamp,cos,cumsum,div,eq,exp,floor,gather,ge,gelu,gt,hardtanh,index_select,le,log,log_softmax,lt,matmul,maximum,mean,minimum,mm,mul,ne,neg,pow,reciprocal,relu,round,rsqrt,sigmoid,sign,silu,sin,softmax,sort,sqrt,sub,sum,tanh,topk
sha=47c1c4969c4312b4f15fc73928c72e4c262f79ce

mkdir -p "$base_host"
: >"$launcher_log"
: >"$monitor_log"
: >"$foreign_flag"
: >"$done_file"

gpu_uuid="$(
  docker exec "$container" "$ixsmi" --query-gpu=index,uuid \
    --format=csv,noheader,nounits |
    awk -F',' -v id="$physical_gpu" '{gsub(/ /, "", $1); if ($1 == id) {gsub(/ /, "", $2); print $2; exit}}'
)"
if [[ -z "$gpu_uuid" ]]; then
  echo "GPU $physical_gpu was not found" >&2
  exit 2
fi

for sample in 1 2 3; do
  status="$(
    docker exec "$container" "$ixsmi" --query-gpu=index,utilization.gpu,memory.used \
      --format=csv,noheader,nounits |
      awk -F',' -v id="$physical_gpu" '{gsub(/ /, "", $1); if ($1 == id) {gsub(/ /, "", $2); gsub(/ /, "", $3); print $2 "," $3; exit}}'
  )"
  util="${status%%,*}"
  memory="${status##*,}"
  processes="$(
    docker exec "$container" "$ixsmi" --query-compute-apps=gpu_uuid,pid \
      --format=csv,noheader,nounits |
      awk -F',' -v uuid="$gpu_uuid" '{gsub(/ /, "", $1); if ($1 == uuid) print $0}' |
      wc -l
  )"
  {
    echo "precheck_sample=$sample util=$util memory_mib=$memory processes=$processes"
  } >>"$launcher_log"
  if (( util > 5 )) || (( memory > 512 )) || (( processes > 0 )); then
    echo "GPU $physical_gpu is not idle (util=$util%, memory=$memory MiB, processes=$processes)" >&2
    exit 2
  fi
  sleep 2
done

monitor() {
  while [[ ! -s "$done_file" ]]; do
    {
      echo "sample=$(date --iso-8601=seconds)"
      rows="$(
        docker exec "$container" "$ixsmi" --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
          --format=csv,noheader,nounits |
          awk -F',' -v uuid="$gpu_uuid" '{gsub(/ /, "", $1); if ($1 == uuid) print $0}'
      )"
      if [[ -z "$rows" ]]; then
        echo "gpu_rows=none"
      else
        while IFS= read -r row; do
          echo "gpu_rows=$row"
          pid="$(echo "$row" | cut -d',' -f2 | tr -d ' ')"
          if [[ ! -e "/proc/$pid" ]]; then
            echo "process_owner=exited_before_attribution pid=$pid"
            continue
          fi
          cgroup="$(tr '\n' ' ' <"/proc/$pid/cgroup" 2>/dev/null || true)"
          cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
          if [[ "$cgroup" == *"$container_id"* ]]; then
            echo "process_owner=benchmark_container pid=$pid cmdline=$cmdline"
          else
            echo "process_owner=foreign pid=$pid cgroup=$cgroup cmdline=$cmdline"
            if [[ ! -s "$foreign_flag" ]]; then
              echo "$pid" >>"$foreign_flag"
              docker exec "$container" pkill -f "$harness" 2>/dev/null || true
            fi
          fi
        done <<<"$rows"
      fi
    } >>"$monitor_log" 2>&1
    sleep 2
  done
}

run_profile() {
  local profile="$1"
  local output="$base/$profile"
  local selected_ops="${2:-}"
  local -a command=(
    python3 "$harness"
    --metadata "$metadata"
    --output-dir "$output"
    --device cuda
    --dtype float16
    --sha "$sha"
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
    -e CUDA_VISIBLE_DEVICES="$physical_gpu" \
    -e INFINIOPS_BENCHMARK_PROFILE="$profile" \
    -e PYTHONPATH="$python_path" \
    -e LD_LIBRARY_PATH="$ld_library_path" \
    "$container" "${command[@]}" 2>&1 | tee -a "$launcher_log"
  return "${PIPESTATUS[0]}"
}

monitor &
monitor_pid=$!

exit_code=0
run_profile decode-4k || exit_code=$?
if [[ "$exit_code" -eq 0 ]]; then
  run_profile decode-8k "$target_ops" || exit_code=$?
fi
if [[ "$exit_code" -eq 0 ]]; then
  run_profile prefill-4k "$target_ops" || exit_code=$?
fi

if [[ "$exit_code" -eq 0 ]]; then
  docker exec \
    -e PYTHONPATH="$python_path" \
    -e LD_LIBRARY_PATH="$ld_library_path" \
    "$container" python3 "$exporter" \
      "$base/decode-4k/results.json" \
      "$base/decode-8k/results.json" \
      "$base/prefill-4k/results.json" \
      --output "$base/infiniops-iluvatar-biv150.csv" \
      --diagnostics "$base/diagnostics.csv" \
      --report "$base/report.md" \
      --platform iluvatar \
      --date "$run_date" \
      --tester baoming 2>&1 | tee -a "$launcher_log"
  exit_code="${PIPESTATUS[0]}"
fi

if [[ -s "$foreign_flag" ]]; then
  foreign_process_seen=1
else
  foreign_process_seen=0
fi
{
  echo "exit_code=$exit_code"
  echo "foreign_process_seen=$foreign_process_seen"
  echo "end=$(date --iso-8601=seconds)"
} >"$done_file"

wait "$monitor_pid"
exit "$exit_code"
