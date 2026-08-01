#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="libjpeg-turbo-rs-opencv-smoke:ubuntu24.04"
LIB=""
WORKDIR=""

usage() {
  echo "usage: $0 --lib <release-cdylib> --workdir <output-dir> [--image <tag>]" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lib)
      [[ $# -ge 2 && -n "$2" ]] || usage
      LIB="${2:-}"
      shift 2
      ;;
    --workdir)
      [[ $# -ge 2 && -n "$2" ]] || usage
      WORKDIR="${2:-}"
      shift 2
      ;;
    --image)
      [[ $# -ge 2 && -n "$2" ]] || usage
      IMAGE="${2:-}"
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

[[ -n "$LIB" && -f "$LIB" && -n "$WORKDIR" ]] || usage
LIB="$(readlink -f "$LIB")"
mkdir -p "$WORKDIR"
WORKDIR="$(readlink -f "$WORKDIR")"
if [[ "$WORKDIR" == "/" ]]; then
  echo "refusing to use the filesystem root as the output directory" >&2
  exit 2
fi

docker build --pull=false --tag "$IMAGE" "$SCRIPT_DIR"
docker run --rm \
  --mount "type=bind,src=$LIB,dst=/input/liblibjpeg_turbo_rs_capi.so,readonly" \
  --mount "type=bind,src=$SCRIPT_DIR,dst=/harness,readonly" \
  --mount "type=bind,src=$WORKDIR,dst=/work" \
  "$IMAGE" bash /harness/container_run.sh
