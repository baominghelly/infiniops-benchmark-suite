#!/usr/bin/env bash
set -euo pipefail

# Ascend245 disallows user crontab, so the A100 scheduler kicks its periodic
# worker over SSH. The worker still owns its per-run lease and heartbeat state.
ssh -o BatchMode=yes -o ConnectTimeout=10 ascend245 '
  set -euo pipefail
  root=/data-aisoft/baoming/infinibench-workers/ascend-910c
  mkdir -p "$root/logs"
  nohup env INFINIBENCH_DEVICE=ascend-910c \
    "$root/bin/periodic_worker.sh" \
    >>"$root/logs/periodic-worker.log" 2>&1 </dev/null &
  echo $! >"$root/logs/periodic-worker-launch.pid"
'
