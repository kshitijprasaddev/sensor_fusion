#!/usr/bin/env python3
"""Download YOLO11n weights and optionally export a local TensorRT FP16 engine."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument('--output', default='/models')
p.add_argument('--engine', action='store_true')
args = p.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
target = out / 'yolo11n.pt'
if not target.exists():
    model = YOLO('yolo11n.pt')
    shutil.copy2(model.ckpt_path, target)
model = YOLO(str(target))
if args.engine:
    # Fixed 640 input, ONNX opset 17 and no embedded NMS work with TensorRT 8.6.
    model.export(format='engine', imgsz=640, half=True, device=0, opset=17,
                 dynamic=False, simplify=False, nms=False, workspace=2)
manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()
            if p.suffix in ('.pt', '.engine', '.onnx')}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
