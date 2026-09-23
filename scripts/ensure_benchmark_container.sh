#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ensure_benchmark_container.sh <container>

Fails if the container does not exist. By default, starts an existing container
whose state is created or exited. Set AUTO_START_CONTAINER=0 to disable startup
and fail on any non-running container.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

container="$1"
auto_start="${AUTO_START_CONTAINER:-1}"

case "$auto_start" in
  0|1) ;;
  *)
    echo "invalid AUTO_START_CONTAINER: $auto_start; expected 0 or 1" >&2
    exit 2
    ;;
esac

if ! docker inspect "$container" >/dev/null 2>&1; then
  echo "container not found: $container" >&2
  exit 2
fi

container_state="$(docker inspect -f '{{.State.Status}}' "$container")"
if [[ "$container_state" == "running" ]]; then
  exit 0
fi

if [[ "$auto_start" != "1" ]]; then
  echo "container is not running: $container (state=$container_state)" >&2
  exit 2
fi

case "$container_state" in
  created|exited)
    echo "benchmark container is not running; starting it: $container (state=$container_state)" >&2
    if ! docker start "$container"; then
      echo "failed to start benchmark container: $container" >&2
      exit 2
    fi
    ;;
  *)
    echo "benchmark container is in a non-runnable state: $container (state=$container_state)" >&2
    exit 2
    ;;
esac

for _ in $(seq 1 30); do
  if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" == "true" ]]; then
    echo "benchmark container started: $container" >&2
    exit 0
  fi
  sleep 1
done

echo "benchmark container did not reach running state: $container (state=$(docker inspect -f '{{.State.Status}}' "$container"))" >&2
exit 2
