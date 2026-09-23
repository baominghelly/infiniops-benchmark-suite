#!/usr/bin/env bash
set -euo pipefail

device="${INFINIBENCH_DEVICE:-metax-c550}"
root="/data-aisoft/baoming/infinibench-workers/${device}"
case "$device" in
  metax-c550) adapter=run-metax-c550-weekly.sh ;;
  nvidia-a100) adapter=run-nvidia-a100-benchmark.sh ;;
  *) echo "unsupported periodic worker device: $device" >&2; exit 2 ;;
esac
mkdir -p "$root/logs"
exec 9>"$root/periodic.lock"
if ! flock -n 9; then
  # A previous cron invocation still owns the device lease.
  exit 0
fi

log="$root/logs/periodic-worker.log"
if [[ "$(stat -c %s "$log" 2>/dev/null || echo 0)" -gt 10485760 ]]; then
  mv -- "$log" "$log.1"
fi

exec >>"$log" 2>&1
printf 'periodic_tick device=%s time=%s\n' "$device" "$(date --iso-8601=seconds)"
exec python3 "$root/bin/periodic_worker.py" \
  --device "$device" \
  --worker "$root/bin/run_platform_worker.py" \
  --adapter "$root/bin/$adapter"
