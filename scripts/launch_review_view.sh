#!/usr/bin/env bash
set -euo pipefail
mode="${1:?Choose camera, lidar or radar}"
shift
python3 /review-source/scripts/review_player.py "$mode" "$@" &
player=$!
rviz2 -d "/review-source/src/group8_fusion/config/review_${mode}.rviz" --ros-args -p use_sim_time:=true &
viewer=$!
trap 'kill "$player" "$viewer" 2>/dev/null || true' EXIT INT TERM
wait -n "$player" "$viewer"
