#!/usr/bin/env bash
set -euo pipefail
output="${1:-/results/validation}"
weights="${FUSION_WEIGHTS:-/models/yolo11n.engine}"
if [[ ! -f "$weights" ]]; then
  echo 'Prepare the TensorRT model first: ./scripts/run.sh prepare --engine' >&2
  exit 1
fi
mkdir -p "$output"
python3 -m pytest -q tests
python3 -m pip check
python3 scripts/environment.py > "$output/environment.json"
python3 scripts/audit_bags.py /data/studentProject /data/studentProject1 --output "$output/dataset_audit.json"
python3 scripts/compare_backends.py /data/studentProject --output "$output/backend_comparison.json"
runs=()
for bag in studentProject studentProject1; do
  for mode in lidar camera_lidar lidar_radar all; do
    destination="$output/${bag}_${mode}"
    args=()
    if [[ "$mode" == all ]]; then args+=(--video); fi
    python3 scripts/evaluate.py "/data/$bag" --modality "$mode" --weights "$weights" \
      --output "$destination" "${args[@]}"
    runs+=("$destination")
  done
done
python3 scripts/summarize_runs.py "${runs[@]}" --output "$output/comparison"
