#!/usr/bin/env bash
set -euo pipefail

# Manual host-side invocations dispatch into the already-running Hermes
# container. Its s6 setup maps the persistent hermes account to UID/GID 1006;
# starting an arbitrary-UID helper container instead breaks OpenSSH passwd
# lookup. Cron executes this script directly as that mapped Hermes account.
if [[ ! -d /opt/data/ssh ]]; then
  exec docker exec -i -u hermes \
    -e TZ="${TZ:-Asia/Shanghai}" \
    -e CAMBRICON_SSH_HOST="${CAMBRICON_SSH_HOST:-libaoming@180.184.148.218}" \
    -e CAMBRICON_SSH_PORT="${CAMBRICON_SSH_PORT:-10000}" \
    -e REMOTE_RESULT_ROOT="${REMOTE_RESULT_ROOT:-/data-aisoft/infinibench_data/ops/cambricon-mlu590}" \
    -e DEST_RESULT_ROOT="${DEST_RESULT_ROOT:-/data-aisoft/infinibench_data/ops/cambricon-mlu590}" \
    -e COLLECTOR_STATE_ROOT="${COLLECTOR_STATE_ROOT:-/data-aisoft/baoming/hermes/runs/cambricon-result-collector}" \
    -e COLLECTOR_DRY_RUN="${COLLECTOR_DRY_RUN:-0}" \
    baoming_hermes_agent /opt/data/scripts/run-cambricon-result-collector.sh
fi

ssh_host="${CAMBRICON_SSH_HOST:-libaoming@180.184.148.218}"
ssh_port="${CAMBRICON_SSH_PORT:-10000}"
remote_root="${REMOTE_RESULT_ROOT:-/data-aisoft/infinibench_data/ops/cambricon-mlu590}"
dest_root="${DEST_RESULT_ROOT:-/data-aisoft/infinibench_data/ops/cambricon-mlu590}"
state_root="${COLLECTOR_STATE_ROOT:-/data-aisoft/baoming/hermes/runs/cambricon-result-collector}"
ssh_key="${CAMBRICON_SSH_KEY:-/opt/data/ssh/cambricon_result_collector}"
known_hosts="${CAMBRICON_KNOWN_HOSTS:-/opt/data/ssh/cambricon_known_hosts}"
dry_run="${COLLECTOR_DRY_RUN:-0}"
stage=""

cleanup() {
  if [[ -n "${stage:-}" && -d "$stage" ]]; then
    rm -rf -- "$stage"
  fi
}
trap cleanup EXIT

for command in ssh tar flock find sha256sum; do
  command -v "$command" >/dev/null || {
    echo "required command is unavailable: $command" >&2
    exit 2
  }
done

for path in "$ssh_key" "$known_hosts"; do
  test -s "$path" || {
    echo "required SSH file is missing or empty: $path" >&2
    exit 2
  }
done

for path in "$remote_root" "$dest_root" "$state_root"; do
  if [[ ! "$path" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
    echo "unsafe absolute path: $path" >&2
    exit 2
  fi
done

mkdir -p "$dest_root" "$state_root/skipped"
exec 9>"$state_root/collector.lock"
if ! flock -n 9; then
  # A previous invocation may still be transferring a large result tree.
  exit 0
fi

run_ssh() {
  ssh \
    -i "$ssh_key" \
    -p "$ssh_port" \
    -o "UserKnownHostsFile=$known_hosts" \
    -o StrictHostKeyChecking=yes \
    -o BatchMode=yes \
    -o ConnectTimeout=10 \
    "$@"
}

if ! candidate_output="$(run_ssh "$ssh_host" \
  "for path in '$remote_root'/*; do if [ -s \"\$path/benchmark.done\" ]; then basename -- \"\$path\"; fi; done | sort")"; then
  echo "cambricon collector cannot inspect remote result root: $remote_root" >&2
  exit 1
fi

record_skip() {
  local run_dir="$1"
  local reason="$2"
  local marker="$state_root/skipped/$run_dir"
  mkdir -p "$state_root/skipped"
  if [[ ! -e "$marker" ]]; then
    printf 'reason=%s\nrecorded_at=%s\n' "$reason" "$(date --iso-8601=seconds)" >"$marker"
    echo "skipped run=$run_dir reason=$reason"
  fi
}

while IFS= read -r run_dir; do
  [[ -z "$run_dir" ]] && continue
  if [[ ! "$run_dir" =~ ^([0-9]{8}|backfill-[0-9]{8}-[0-9]{8})$ ]]; then
    record_skip "$run_dir" invalid_run_directory_name
    continue
  fi

  final="$dest_root/$run_dir"
  if [[ -d "$final" && -s "$final/.collector-success.json" ]]; then
    continue
  fi

  remote_csv="$(run_ssh "$ssh_host" \
    "find '$remote_root/$run_dir' -maxdepth 1 -type f -name 'infiniops-*.csv' -size +0c -print -quit" || true)"
  if [[ -z "$remote_csv" ]]; then
    record_skip "$run_dir" completed_without_result_csv
    continue
  fi

  if [[ "$dry_run" == "1" ]]; then
    echo "would_collect run=$run_dir remote_csv=$remote_csv final=$final"
    continue
  fi

  if [[ -e "$final" ]]; then
    record_skip "$run_dir" destination_already_exists
    continue
  fi

  stage="$(mktemp -d "$dest_root/.collector-staging-${run_dir}.XXXXXX")"
  if [[ ! "$stage" =~ ^$dest_root/\.collector-staging-[0-9a-zA-Z.-]+\.XXXXXX$ && ! "$stage" =~ ^$dest_root/\.collector-staging-[0-9a-zA-Z.-]+\.[A-Za-z0-9]+$ ]]; then
    echo "refusing unexpected staging path: $stage" >&2
    exit 2
  fi

  run_ssh "$ssh_host" "tar -C '$remote_root/$run_dir' -cf - ." | tar -C "$stage" -xf -

  test -s "$stage/benchmark.done" || {
    echo "copied run is missing completion metadata: $run_dir" >&2
    exit 1
  }
  csv_count="$(find "$stage" -maxdepth 1 -type f -name 'infiniops-*.csv' -size +0c | wc -l)"
  [[ "$csv_count" -gt 0 ]] || {
    echo "copied run has no non-empty result CSV: $run_dir" >&2
    exit 1
  }

  (
    cd "$stage"
    find . -type f ! -name '.collector-manifest.sha256' -print0 \
      | sort -z | xargs -0 sha256sum
  ) >"$stage/.collector-manifest.sha256"

  file_count="$(find "$stage" -type f | wc -l)"
  byte_count="$(find "$stage" -type f -printf '%s\n' | awk '{sum += $1} END {printf "%.0f", sum}')"
  cat >"$stage/.collector-success.json" <<EOF
{
  "status": "success",
  "platform": "cambricon-mlu590",
  "run_dir": "$run_dir",
  "source_host": "$ssh_host",
  "source_root": "$remote_root",
  "collected_at": "$(date --iso-8601=seconds)",
  "file_count": $file_count,
  "byte_count": $byte_count
}
EOF

  mv "$stage" "$final"
  stage=""
  echo "collected run=$run_dir files=$file_count bytes=$byte_count final=$final"
done <<<"$candidate_output"
