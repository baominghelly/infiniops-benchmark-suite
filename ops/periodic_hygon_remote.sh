#!/usr/bin/env bash
set -uo pipefail

# BW1100 nodes disallow user crontab, so A100 kicks their periodic workers over
# SSH. Both nodes use the logical hygon-bw1100 platform and shared worker lease;
# if one node is healthy, the other exits when it observes its fresh heartbeat.
launch_node() {
  local host="$1" node="$2" root="$3" container="$4"
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$host" \
    "set -euo pipefail
     root='$root'
     node='$node'
     container='$container'
     mkdir -p \"\$root/logs\"
     nohup env INFINIBENCH_DEVICE=hygon-bw1100 \\
       INFINIBENCH_NODE=\"\$node\" \\
       INFINIBENCH_WORKER_ROOT=\"\$root\" \\
       INFINIBENCH_CONTAINER=\"\$container\" \\
       \"\$root/bin/periodic_worker.sh\" \\
       >>\"\$root/logs/periodic-worker-launch.log\" 2>&1 </dev/null &
     echo \$! >\"\$root/logs/periodic-worker-launch.pid\""
}

launch_node hygon251 hygon-bw1100-01 \
  /data-aisoft/baoming/infinibench-workers/hygon-bw1100-01 \
  baoming_infiniops_hygon || echo "hygon-bw1100-01 periodic launch failed" >&2
launch_node hygon252 hygon-bw1100-02 \
  /data-aisoft/baoming/infinibench-workers/hygon-bw1100-02 \
  baoming_infiniops_hygon_dcu2 || echo "hygon-bw1100-02 periodic launch failed" >&2
