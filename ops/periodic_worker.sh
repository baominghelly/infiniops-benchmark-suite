#!/usr/bin/env bash
set -euo pipefail

device="${INFINIBENCH_DEVICE:-metax-c550}"
root="${INFINIBENCH_WORKER_ROOT:-/data-aisoft/baoming/infinibench-workers/$device}"
node="${INFINIBENCH_NODE:-}"

case "$device" in
  metax-c550) adapter=run-metax-c550-weekly.sh ;;
  nvidia-a100) adapter=run-nvidia-a100-benchmark.sh ;;
  ascend-910c) adapter=run-ascend-910c.sh ;;
  iluvatar-tg150) adapter=run-iluvatar-tg150.sh ;;
  moore-s5000) adapter=run-moore-s5000.sh ;;
  cambricon-mlu590) adapter=run-cambricon-mlu590.sh ;;
  hygon-bw1100)
    adapter=run-hygon-bw1100.sh
    if [[ -z "$node" ]]; then
      case "$(hostname -s)" in
        dcu1) node=hygon-bw1100-01 ;;
        dcu2) node=hygon-bw1100-02 ;;
        *) node="$(hostname -s)" ;;
      esac
    fi
    export INFINIBENCH_NODE="$node"
    ;;
  *) echo "unsupported periodic worker device: $device" >&2; exit 2 ;;
esac

node_args=()
if [[ -n "$node" ]]; then
  node_args=(--node-id "$node")
fi

mkdir -p "$root/logs"
exec 9>"$root/periodic.lock"
if ! flock -n 9; then
  # A previous invocation on this installation still owns the node lease.
  exit 0
fi

log="$root/logs/periodic-worker.log"
if [[ "$(stat -c %s "$log" 2>/dev/null || echo 0)" -gt 10485760 ]]; then
  mv -- "$log" "$log.1"
fi

exec >>"$log" 2>&1
printf 'periodic_tick device=%s node=%s time=%s\n' "$device" "${node:-none}" "$(date --iso-8601=seconds)"
exec python3 "$root/bin/periodic_worker.py" \
  --device "$device" \
  "${node_args[@]}" \
  --worker "$root/bin/run_platform_worker.py" \
  --adapter "$root/bin/$adapter"
