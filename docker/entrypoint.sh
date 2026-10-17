#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
source /workspace/sensor_fusion/install/setup.bash
exec /opt/nvidia/nvidia_entrypoint.sh "$@"
