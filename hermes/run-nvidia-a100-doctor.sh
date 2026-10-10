#!/usr/bin/env bash
set -euo pipefail

exec python3 /opt/data/scripts/nvidia_a100_doctor.py \
  --runs-root /data-aisoft/baoming/hermes/runs \
  --container baoming_infiniops \
  --workflow /opt/data/scripts/run-nvidia-a100-benchmark.sh \
  --apply \
  --execute-retry
