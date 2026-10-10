#!/usr/bin/env bash
set -euo pipefail

print_usage() {
  cat <<'EOF'
Usage:
  run_benchmark.sh <device> [platform-runner-args...]
  DEVICE=<device> run_benchmark.sh [platform-runner-args...]

The first non-option argument is the device when DEVICE is not set. Remaining
arguments are forwarded to the platform runner, so an explicit GPU index can
still be passed as the next argument.

Supported devices:
  nvidia-a100      aliases: nvidia, a100
  metax-c550       aliases: metax, c550
  ascend-910c      aliases: ascend, 910c
  moore-s5000      aliases: moore
  iluvatar-tg150   aliases: iluvatar, tianshu
  cambricon-mlu590 aliases: cambricon, mlu
  hygon-bw1000     aliases: hygon, dcu
  hygon-bw1100     aliases: bw1100, bw1100-01, bw1100-02

Examples:
  run_benchmark.sh metax-c550
  run_benchmark.sh nvidia-a100 3
  DEVICE=moore-s5000 BACKFILL_DATE=2026-09-04 run_benchmark.sh
EOF
}

print_devices() {
  cat <<'EOF'
nvidia-a100
metax-c550
ascend-910c
moore-s5000
iluvatar-tg150
cambricon-mlu590
hygon-bw1000
hygon-bw1100
EOF
}

requested_device="${DEVICE:-}"
print_config=0
forward_args=()

if [[ $# -gt 0 ]]; then
  case "$1" in
    --device)
      if [[ "$#" -lt 2 ]]; then
        echo "--device requires one device value" >&2
        exit 2
      fi
      requested_device="$2"
      shift 2
      ;;
    --list)
      print_devices
      exit 0
      ;;
    --print)
      print_config=1
      shift
      if [[ -z "$requested_device" && "$#" -gt 0 && "$1" != -* ]]; then
        requested_device="$1"
        shift
      fi
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    --)
      shift
      ;;
    -*)
      echo "unknown option: $1" >&2
      print_usage >&2
      exit 2
      ;;
    *)
      if [[ -z "$requested_device" ]]; then
        requested_device="$1"
        shift
      fi
      ;;
  esac
fi
forward_args+=("$@")

if [[ -z "$requested_device" ]]; then
  print_usage >&2
  exit 2
fi

case "$requested_device" in
  nvidia|a100|nvidia-a100)
    device_id=nvidia-a100
    runner=run_a100_benchmark.sh
    ;;
  metax|c550|metax-c550)
    device_id=metax-c550
    runner=run_metax_benchmark.sh
    ;;
  ascend|910c|ascend-910c)
    device_id=ascend-910c
    runner=run_ascend_benchmark.sh
    ;;
  moore|moore-s5000)
    device_id=moore-s5000
    runner=run_moore_benchmark.sh
    ;;
  iluvatar|tianshu|iluvatar-tg150)
    device_id=iluvatar-tg150
    runner=run_iluvatar_benchmark.sh
    ;;
  cambricon|mlu|cambricon-mlu590)
    device_id=cambricon-mlu590
    runner=run_cambricon_benchmark.sh
    ;;
  hygon|dcu|hygon-bw1000)
    device_id=hygon-bw1000
    runner=run_hygon_benchmark.sh
    ;;
  bw1100|hygon-bw1100)
    device_id=hygon-bw1100
    runner=run_hygon_benchmark.sh
    ;;
  bw1100-01|hygon-bw1100-01|bw1100-02|hygon-bw1100-02)
    # Node suffixes are physical execution hints, not separate result platforms.
    device_id=hygon-bw1100
    runner=run_hygon_benchmark.sh
    case "$requested_device" in
      *-01) hygon_node_hint=hygon-bw1100-01 ;;
      *-02) hygon_node_hint=hygon-bw1100-02 ;;
    esac
    ;;
  *)
    echo "unknown device: $requested_device" >&2
    echo "run with --list to see supported devices" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runner_path="$script_dir/$runner"
output_root="${OUTPUT_ROOT:-/data-aisoft/infinibench_data/ops}"
if [[ "$device_id" == "hygon-bw1100" ]]; then
  if [[ -z "${INFINIBENCH_NODE:-}" ]]; then
    case "$(hostname -s)" in
      dcu1) export INFINIBENCH_NODE=hygon-bw1100-01 ;;
      dcu2) export INFINIBENCH_NODE=hygon-bw1100-02 ;;
      *) export INFINIBENCH_NODE="$(hostname -s)" ;;
    esac
  fi
  if [[ -n "${hygon_node_hint:-}" ]]; then
    export INFINIBENCH_NODE="$hygon_node_hint"
  fi
  if [[ "$INFINIBENCH_NODE" == "hygon-bw1100-02" && -z "${CONTAINER:-}" ]]; then
    export CONTAINER=baoming_infiniops_hygon_dcu2
    export INFINIBENCH_CONTAINER="$CONTAINER"
  fi
fi

if [[ "$print_config" == "1" ]]; then
  printf 'DEVICE_ID=%s\n' "$device_id"
  printf 'RUNNER=%s\n' "$runner_path"
  printf 'OUTPUT_ROOT=%s\n' "$output_root"
  printf 'OUTPUT_DIR=%s/%s/<run-date>\n' "$output_root" "$device_id"
  exit 0
fi

if [[ ! -x "$runner_path" ]]; then
  echo "platform runner not found or not executable: $runner_path" >&2
  exit 2
fi

# These are the shared delivery contracts. Platform runners keep their own
# source, container, runtime, build-cache, and GPU-monitoring defaults.
export DEVICE_ID="$device_id"
export OUTPUT_ROOT="$output_root"

exec "$runner_path" "${forward_args[@]}"
