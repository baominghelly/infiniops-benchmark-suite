#!/usr/bin/env bash
set -uo pipefail

container=baoming_infiniops
physical_gpu="${1:?physical GPU index is required}"
sha=7be8ab723c5229a8af97d7dbca803b2e0422a157
base=/data/shared/baoming/InfiniOps_benchmark_results/2026-08-30
tools_root=/data/shared/baoming/InfiniOps_benchmark_tools
harness="$tools_root/scripts/full_operator_benchmark.py"
exporter="$tools_root/scripts/export_full_operator_report.py"
metadata=/usr/local/lib/python3.12/dist-packages/infini/torch_ops_metadata.json
launcher_log="$base/benchmark-launcher.log"
monitor_log="$base/benchmark-monitor.log"
foreign_flag="$base/foreign-process.flag"
done_file="$base/benchmark.done"
container_id="$(docker inspect --format '{{.Id}}' "$container")"
gpu_uuid="$(nvidia-smi -i "$physical_gpu" --query-gpu=uuid --format=csv,noheader,nounits)"
target_ops=abs,add,addmm,amax,amin,argmax,bitwise_and,bitwise_not,bitwise_or,bmm,cat,ceil,clamp,cos,cumsum,div,eq,exp,floor,gather,ge,gelu,gt,hardtanh,index_select,le,log,log_softmax,lt,matmul,maximum,mean,minimum,mm,mul,ne,neg,pow,reciprocal,relu,round,rsqrt,sigmoid,sign,silu,sin,softmax,sort,sqrt,sub,sum,tanh,topk

mkdir -p "$base"
: >"$launcher_log"
: >"$monitor_log"
: >"$foreign_flag"
: >"$done_file"

{
  echo "start=$(date --iso-8601=seconds)"
  echo "physical_gpu=$physical_gpu"
  echo "gpu_uuid=$gpu_uuid"
  echo "container=$container"
  echo "container_id=$container_id"
  echo "sha=$sha"
  echo "metadata=$metadata"
  echo "profiles=decode-4k(full),decode-8k(targeted),prefill-4k(targeted)"
  echo "target_ops=$target_ops"
  echo "=== pre-run nvidia-smi ==="
  nvidia-smi -i "$physical_gpu"
} >>"$launcher_log" 2>&1

monitor() {
  while [[ ! -s "$done_file" ]]; do
    {
      echo "sample=$(date --iso-8601=seconds)"
      rows="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits | awk -F ', ' -v uuid="$gpu_uuid" '$1 == uuid {print $2 ", " $3 ", " $4}')"
      if [[ -z "$rows" ]]; then
        echo "gpu_rows=none"
      else
        while IFS= read -r row; do
          echo "gpu_rows=$row"
          pid="${row%%,*}"
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
  docker exec "$container" python3 "$exporter" \
    "$base/decode-4k/results.json" \
    "$base/decode-8k/results.json" \
    "$base/prefill-4k/results.json" \
    --output "$base/infiniops-nvidia-a100.csv" \
    --diagnostics "$base/diagnostics.csv" \
    --report "$base/report.md" \
    --platform nvidia \
    --date 2026/08/30 \
    --tester baoming 2>&1 | tee -a "$launcher_log"
  exit_code=${PIPESTATUS[0]}
fi

{
  echo "=== post-run nvidia-smi ==="
  nvidia-smi -i "$physical_gpu"
  echo "end=$(date --iso-8601=seconds)"
  echo "exit_code=$exit_code"
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
} >"$done_file"

wait "$monitor_pid"
exit "$exit_code"
