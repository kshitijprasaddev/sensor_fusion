#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-fusion}"
if [[ $# -gt 0 ]]; then shift; fi
root="$PWD/results/review_2026_11_26"
mkdir -p "$root"
if [[ "$mode" == fusion ]]; then
  cp src/group8_fusion/config/fusion_review.yaml "$root/fusion_review.yaml"
  exec ./scripts/run.sh ai-demo config:=/results/review_2026_11_26/fusion_review.yaml "$@"
fi
if [[ "$mode" != prepare && "$mode" != camera && "$mode" != lidar && "$mode" != radar ]]; then
  echo 'Usage: scripts/review_demo.sh {prepare|camera|lidar|radar|fusion} [arguments]' >&2
  exit 2
fi
image="${FUSION_IMAGE:-group8-sensor-fusion:ai}"
data="${DATA_DIR:-/home/kshitij0/Downloads/studentProject1}"
args=(--rm --init --ipc=host --user "$(id -u):$(id -g)" -e HOME=/tmp/group8
  -e YOLO_CONFIG_DIR=/tmp/ultralytics -e ROS_DOMAIN_ID=38
  -v "$data:/data:ro" -v "$PWD/models:/models:ro" -v "$PWD/results:/results"
  -v "$PWD:/review-source:ro")
gpu_args_file=$(mktemp)
trap 'rm -f "$gpu_args_file"' EXIT
python3 scripts/gpu_args.py > "$gpu_args_file"
mapfile -d '' -t review_gpu_args < "$gpu_args_file"
rm -f "$gpu_args_file"
trap - EXIT
if [[ "$mode" == prepare ]]; then
  exec docker run "${args[@]}" "${review_gpu_args[@]}" "$image" python3 /review-source/scripts/prepare_review.py "$@"
fi
if [[ ! -f "$root/manifest.json" ]]; then
  echo 'First run ./scripts/review_demo.sh prepare to compute the diagnostic outputs.' >&2
  exit 1
fi
args+=(--name group8-review -e "DISPLAY=${DISPLAY:?Run from your graphical desktop terminal}"
  -e NVIDIA_DRIVER_CAPABILITIES=all -v /tmp/.X11-unix:/tmp/.X11-unix:rw)
if [[ -f "${XAUTHORITY:-}" ]]; then
  args+=(-e XAUTHORITY=/tmp/host.xauth -v "$XAUTHORITY:/tmp/host.xauth:ro")
fi
exec docker run "${args[@]}" "${review_gpu_args[@]}" "$image" bash /review-source/scripts/launch_review_view.sh "$mode" "$@"
