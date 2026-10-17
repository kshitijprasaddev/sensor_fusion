#!/usr/bin/env bash
set -e
mkdir -p "${HOME:-/tmp/group8}" "${YOLO_CONFIG_DIR:-/tmp/ultralytics}" /tmp/group8-xdg
chmod 700 /tmp/group8-xdg
export XDG_RUNTIME_DIR=/tmp/group8-xdg
source /opt/ros/humble/setup.bash
source /workspace/sensor_fusion/install/setup.bash
exec /opt/nvidia/nvidia_entrypoint.sh "$@"
