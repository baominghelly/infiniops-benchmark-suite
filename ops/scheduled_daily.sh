#!/usr/bin/env bash
set -euo pipefail

export TZ=Asia/Shanghai
state_root=/data-aisoft/baoming/infinibench-ops
mkdir -p "$state_root/logs"
exec 9>"$state_root/.daily-schedule.lock"
if ! flock -n 9; then
  echo "another daily launcher is active" >&2
  exit 0
fi

today="$(date +%Y%m%d)"
exec /data-aisoft/baoming/infinibench-ops/bin/infinibenchctl \
  run-once \
  --device nvidia-a100 \
  --device metax-c550 \
  --global-run-id "${today}-daily" \
  --execution-time "${today}-020000" \
  --schedule-mode daily \
  --dtypes float16,float32,bfloat16 \
  --profile decode-4k \
  --infiniops-ref master \
  --infinirt-ref master
