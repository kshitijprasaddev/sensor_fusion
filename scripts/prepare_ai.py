#!/usr/bin/env python3
"""Fetch published perception checkpoints, recording sources and hashes."""
import hashlib
import json
from pathlib import Path
import gdown
from ultralytics import SAM

out = Path("/models")
out.mkdir(exist_ok=True)
checkpoint = out / "voxelnext_nuscenes.pth"
if not checkpoint.exists():
    temp = checkpoint.with_suffix(".partial")
    result = gdown.download(
        id="1IV7e7G9X-61KXSjMGtQo579pzDNbhwvf", output=str(temp), quiet=False
    )
    if not result or temp.stat().st_size < 1_000_000:
        raise RuntimeError("Incomplete VoxelNeXt checkpoint")
    temp.rename(checkpoint)
SAM(str(out / "sam2.1_t.pt"))
items = {}
for name, source in [
    ("voxelnext_nuscenes.pth", "https://github.com/JIA-Lab-research/VoxelNeXt"),
    ("sam2.1_t.pt", "https://github.com/ultralytics/assets/releases"),
]:
    file = out / name
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    expected = {
        "voxelnext_nuscenes.pth": "9409dd8c13c8c8ca546c8c5af024856d03029e2def8d5fc0fa3bfe4477e7d88b",
        "sam2.1_t.pt": "3c1e81ca9b037dd39d70a014ddb9a813d6c4c4e12555420db7eaff31689bd4e3",
    }
    if digest != expected[name]:
        raise ValueError(f"Checkpoint hash mismatch: {name}")
    items[name] = {
        "sha256": digest,
        "bytes": file.stat().st_size,
        "source": source,
        "training": "upstream pretrained; no local ground-truth fine-tuning",
    }
(out / "ai_manifest.json").write_text(json.dumps(items, indent=2) + "\n")
print(json.dumps(items, indent=2))
