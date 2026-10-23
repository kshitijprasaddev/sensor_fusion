#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
image="${FUSION_IMAGE:-group8-sensor-fusion:humble}"
command="${1:-demo}"
if [[ $# -gt 0 ]]; then shift; fi
ai_mode=0
mid_mode=0
birdseye=0
if [[ "$command" == mid-* ]]; then
  ai_mode=1
  mid_mode=1
  image="${FUSION_IMAGE:-group8-sensor-fusion:ai}"
  command="${command#mid-}"
elif [[ "$command" == ai-* ]]; then
  ai_mode=1
  image="${FUSION_IMAGE:-group8-sensor-fusion:ai}"
  command="${command#ai-}"
fi
if [[ "$command" == birdseye && "$ai_mode" == 1 && "$mid_mode" == 0 ]]; then
  birdseye=1
  command=demo
fi
if [[ "$command" == build ]]; then
  if [[ "$ai_mode" == 1 ]]; then
    exec docker build --progress=plain -f Dockerfile.ai -t "$image" . "$@"
  fi
  exec docker build --progress=plain -t "$image" . "$@"
fi
if [[ "$mid_mode" == 1 && "$command" == demo ]]; then
  set -- config:=/workspace/sensor_fusion/src/group8_fusion/config/fusion_middle.yaml \
    weights:=/models/yolo11n.engine rate:=0.1 "$@"
elif [[ "$mid_mode" == 1 && "$command" == evaluate ]]; then
  set -- --config src/group8_fusion/config/fusion_middle.yaml "$@"
elif [[ "$ai_mode" == 1 && "$command" == demo ]]; then
  set -- config:=/workspace/sensor_fusion/src/group8_fusion/config/fusion_ai.yaml \
    weights:=/models/yolo11n.engine rate:=0.1 "$@"
elif [[ "$ai_mode" == 1 && "$command" == evaluate ]]; then
  set -- --config src/group8_fusion/config/fusion_ai.yaml "$@"
fi
data="${DATA_DIR:-/home/kshitij0/Downloads/studentProject1}"
if [[ ! -d "$data" ]]; then echo "Set DATA_DIR to the directory containing studentProject and studentProject1" >&2; exit 1; fi
mkdir -p models results
args=(--rm --init --ipc=host --shm-size=2g --user "$(id -u):$(id -g)" -e HOME=/tmp/group8
  -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-8}"
  -e YOLO_CONFIG_DIR=/tmp/ultralytics
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
  prepare)
    if [[ "$ai_mode" == 1 ]]; then
      exec docker run "${args[@]}" "$image" python3 scripts/prepare_ai.py "$@"
    fi
    exec docker run "${args[@]}" "$image" python3 scripts/prepare_model.py "$@" ;;
  extract)
    if [[ "$mid_mode" == 1 ]]; then
      exec docker run "${args[@]}" "$image" python3 scripts/extract_mid.py "$@"
    fi
    exec docker run "${args[@]}" "$image" python3 scripts/extract_ai.py "$@" ;;
  train)
    if [[ "$mid_mode" == 1 ]]; then
      exec docker run "${args[@]}" "$image" python3 scripts/train_mid.py "$@"
    fi
    exec docker run "${args[@]}" "$image" python3 scripts/train_ai_v2.py "$@" ;;
  check)
    if [[ "$mid_mode" == 1 ]]; then
      exec docker run "${args[@]}" "$image" python3 scripts/verify_middle.py "$@"
    fi
    echo 'Use mid-check for the middle-fusion runtime guard.' >&2
    exit 1 ;;
  train-v1) exec docker run "${args[@]}" "$image" python3 scripts/train_ai.py "$@" ;;
  audit) exec docker run "${args[@]}" "$image" python3 scripts/audit_bags.py /data/studentProject /data/studentProject1 --output /results/dataset_audit.json "$@" ;;
  evaluate) exec docker run "${args[@]}" "$image" python3 scripts/evaluate.py "$@" ;;
  test) exec docker run "${args[@]}" "$image" python3 -m pytest -q tests "$@" ;;
  validate)
    if [[ "$mid_mode" == 1 ]]; then
      exec docker run "${args[@]}" "$image" bash scripts/validate_mid.sh "$@"
    fi
    if [[ "$ai_mode" == 1 ]]; then
      exec docker run "${args[@]}" "$image" bash scripts/validate_ai.sh "$@"
    fi
    exec docker run "${args[@]}" "$image" bash scripts/validate_all.sh "$@" ;;
  calibrate)
    calibration_file=$(realpath "${1:?Provide the ROS camera calibration YAML path}")
    shift
    exec docker run "${args[@]}" -v "$calibration_file:/tmp/camera.yaml:ro" "$image" \
      python3 scripts/import_calibration.py /tmp/camera.yaml --output /results/fusion_calibrated.yaml "$@" ;;
  demo)
    if [[ -z "${DISPLAY:-}" ]]; then echo 'DISPLAY is required for RViz. Use shell or headless launch otherwise.' >&2; exit 1; fi
    args+=(--name "${FUSION_CONTAINER_NAME:-group8-rviz}" -e "DISPLAY=$DISPLAY" -e NVIDIA_DRIVER_CAPABILITIES=all -v /tmp/.X11-unix:/tmp/.X11-unix:rw)
    if [[ -f "${XAUTHORITY:-}" ]]; then args+=(-e XAUTHORITY=/tmp/host.xauth -v "$XAUTHORITY:/tmp/host.xauth:ro"); fi
    rviz_source=fusion.rviz
    rviz_preferences=fusion.rviz
    if [[ "$mid_mode" == 1 ]]; then
      rviz_source=fusion_middle.rviz
      rviz_preferences=fusion_middle_masks.rviz
    elif [[ "$ai_mode" == 1 ]]; then
      rviz_source=fusion_ai.rviz
      rviz_preferences=fusion_ai_masks.rviz
      if [[ "$birdseye" == 1 ]]; then
        rviz_source=fusion_ai_birdseye.rviz
        rviz_preferences=fusion_ai_masks_birdseye.rviz
      fi
    fi
    if [[ ! -f "results/$rviz_preferences" ]]; then cp "src/group8_fusion/config/$rviz_source" "results/$rviz_preferences"; fi
    exec docker run "${args[@]}" "$image" ros2 launch group8_fusion demo.launch.py "rviz_config:=/results/$rviz_preferences" "$@" ;;
  shell) exec docker run -it "${args[@]}" "$image" bash ;;
  *) echo "Usage: $0 {build|prepare|audit|evaluate|test|validate|calibrate|demo|shell|ai-build|ai-prepare|ai-extract|ai-train|ai-train-v1|ai-evaluate|ai-demo|ai-birdseye|ai-test|ai-validate|ai-shell|mid-build|mid-extract|mid-train|mid-evaluate|mid-demo|mid-check|mid-test|mid-validate|mid-shell} [arguments]" >&2; exit 1 ;;
esac
