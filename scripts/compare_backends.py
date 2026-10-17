#!/usr/bin/env python3
"""Check FP16 export agreement on real frames, without claiming accuracy."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from group8_fusion.bag import Bag, CAMERA
from group8_fusion.geometry import image_array, iou
from group8_fusion.processor import load_config, CameraDetector

p = argparse.ArgumentParser()
p.add_argument('bag')
p.add_argument('--config', default='src/group8_fusion/config/fusion.yaml')
p.add_argument('--samples', type=int, default=20)
p.add_argument('--output', default='/results/backend_comparison.json')
args = p.parse_args()
cfg = load_config(args.config)['model']
detectors = [CameraDetector(dict(cfg, weights=f'/models/yolo11n.{ext}')) for ext in ('pt', 'engine')]
bag = Bag(args.bag)
overlaps, timings = [], [[], []]
counts = [0, 0]
matched = 0
for i in np.linspace(0, len(bag.index[CAMERA]) - 1, min(args.samples, len(bag.index[CAMERA])), dtype=int):
    image = image_array(bag.read(CAMERA, int(i))[1])
    boxes = []
    for j, detector in enumerate(detectors):
        start = time.perf_counter()
        boxes.append(detector.detect(image))
        timings[j].append((time.perf_counter() - start) * 1000)
        counts[j] += len(boxes[-1])
    if boxes[0] and boxes[1]:
        scores = np.array([[iou(np.array(a['xyxy']), np.array(b['xyxy'])) if a['label'] == b['label'] else 0
                            for b in boxes[1]] for a in boxes[0]])
        rows, cols = linear_sum_assignment(-scores)
        for row, col in zip(rows, cols):
            if scores[row, col] >= .5:
                overlaps.append(float(scores[row, col]))
                matched += 1
report = {'samples': len(timings[0]), 'pytorch_boxes': counts[0], 'tensorrt_boxes': counts[1],
          'same_class_matches_iou_ge_0_5': matched,
          'matched_iou_median': float(np.median(overlaps)) if overlaps else None,
          'camera_latency_median_ms': dict(zip(['pytorch', 'tensorrt'], map(lambda t: float(np.median(t)), timings))),
          'interpretation': 'Backend agreement only. Neither backend is ground truth.'}
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + '\n')
bag.close()
print(json.dumps(report, indent=2))
