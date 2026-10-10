#!/usr/bin/env bash
set -euo pipefail

name=baoming_hermes_agent
image=infinibench/hermes-agent:ca29a769

if docker ps -aq --filter "name=^${name}$" | grep -q .; then
  docker rm -f "$name"
fi

docker run -d \
  --name "$name" \
  --restart unless-stopped \
  --network bridge \
  --health-cmd "/command/s6-svstat /run/service/gateway-default | grep -q '^up'" \
  --health-interval 30s \
  --health-timeout 5s \
  --health-start-period 30s \
  --health-retries 3 \
  -e TZ=Asia/Shanghai \
  -e HERMES_UID=1006 \
  -e HERMES_GID=1006 \
  -e HERMES_HOME=/opt/data \
  -e HERMES_WRITE_SAFE_ROOT=/opt/data \
  -e HERMES_DISABLE_LAZY_INSTALLS=1 \
  -e HERMES_LAZY_INSTALL_TARGET=/opt/data/lazy-packages \
  -e API_SERVER_ENABLED=false \
  -e HERMES_DASHBOARD=0 \
  -e NVIDIA_VISIBLE_DEVICES=void \
  -e CUDA_VISIBLE_DEVICES= \
  -v /data-aisoft/baoming/hermes/data:/opt/data \
  -v /data-aisoft/baoming/hermes/workspace:/workspace \
  -v /data-aisoft/baoming/hermes/skills:/skills:ro \
  -v /data-aisoft/baoming/hermes/repo:/repo:ro \
  -v /data-aisoft/baoming/hermes/runs:/data-aisoft/baoming/hermes/runs \
  -v /data-aisoft/infinibench-scripts:/data-aisoft/infinibench-scripts:ro \
  -v /data-aisoft/infinibench_data:/data-aisoft/infinibench_data \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro \
  "$image" gateway run --replace
