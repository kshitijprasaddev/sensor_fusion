#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
image="${FUSION_IMAGE:-group8-sensor-fusion:humble}"
command="${1:-demo}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$command" == build ]]; then
  exec docker build --progress=plain -t "$image" . "$@"
fi
data="${DATA_DIR:-/home/kshitij0/Downloads/studentProject1}"
if [[ ! -d "$data" ]]; then echo "Set DATA_DIR to the directory containing studentProject and studentProject1" >&2; exit 1; fi
mkdir -p models results
args=(--rm --init --ipc=host --shm-size=2g
  -v "$data:/data:ro" -v "$PWD/models:/models" -v "$PWD/results:/results")
if [[ "${GPU:-1}" == 1 ]]; then
  gpu_args_file=$(mktemp)
  trap 'rm -f "$gpu_args_file"' EXIT
  python3 scripts/gpu_args.py > "$gpu_args_file"
  mapfile -d '' -t gpu_args < "$gpu_args_file"
  rm -f "$gpu_args_file"
  trap - EXIT
  args+=("${gpu_args[@]}")
fi
case "$command" in
  prepare) exec docker run "${args[@]}" "$image" python3 scripts/prepare_model.py "$@" ;;
  audit) exec docker run "${args[@]}" "$image" python3 scripts/audit_bags.py /data/studentProject /data/studentProject1 --output /results/dataset_audit.json "$@" ;;
  evaluate) exec docker run "${args[@]}" "$image" python3 scripts/evaluate.py "$@" ;;
  test) exec docker run "${args[@]}" "$image" python3 -m pytest -q tests "$@" ;;
  calibrate)
    calibration_file=$(realpath "${1:?Provide the ROS camera calibration YAML path}")
    shift
    exec docker run "${args[@]}" -v "$calibration_file:/tmp/camera.yaml:ro" "$image" \
      python3 scripts/import_calibration.py /tmp/camera.yaml --output /results/fusion_calibrated.yaml "$@" ;;
  demo)
    if [[ -z "${DISPLAY:-}" ]]; then echo 'DISPLAY is required for RViz. Use shell or headless launch otherwise.' >&2; exit 1; fi
    args+=(-e "DISPLAY=$DISPLAY" -e NVIDIA_DRIVER_CAPABILITIES=all -v /tmp/.X11-unix:/tmp/.X11-unix:rw)
    if [[ -f "${XAUTHORITY:-}" ]]; then args+=(-e XAUTHORITY=/tmp/host.xauth -v "$XAUTHORITY:/tmp/host.xauth:ro"); fi
    exec docker run "${args[@]}" "$image" ros2 launch group8_fusion demo.launch.py "$@" ;;
  shell) exec docker run -it "${args[@]}" "$image" bash ;;
  *) echo "Usage: $0 {build|prepare|audit|evaluate|test|calibrate|demo|shell} [arguments]" >&2; exit 1 ;;
esac
