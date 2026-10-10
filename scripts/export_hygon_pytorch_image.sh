#!/usr/bin/env bash
set -euo pipefail

container="${1:?usage: export_hygon_pytorch_image.sh CONTAINER}"

docker export "$container" | zstd -1 -T0
