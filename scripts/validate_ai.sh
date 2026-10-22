#!/usr/bin/env bash
set -euo pipefail
output="${1:-/results/ai_validation}"
mkdir -p "$output"
python3 -m pytest -q tests
python3 -m pip check
python3 scripts/verify_neural.py --output "$output/runtime_guard.json"
python3 scripts/environment.py > "$output/environment.json"
for bag in studentProject studentProject1; do
  python3 scripts/evaluate.py "/data/$bag" --config src/group8_fusion/config/fusion_ai.yaml \
    --weights /models/yolo11n.engine --output "$output/$bag" --video
done
