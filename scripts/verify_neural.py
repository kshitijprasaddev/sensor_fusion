#!/usr/bin/env python3
"""GPU integration test: fail if neural inference calls the classical core."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from group8_fusion import perception, tracking, processor
from group8_fusion.bag import Bag, CAMERA, LIDAR, RADAR
from group8_fusion.geometry import cloud_array, image_array
from group8_fusion.processor import load_config, CameraDetector, make_processor

p = argparse.ArgumentParser()
p.add_argument("--frames", type=int, default=5)
p.add_argument("--output", default="/results/ai_runtime_guard.json")
a = p.parse_args()


def forbidden(*args, **kwargs):
    raise AssertionError(
        "Classical DBSCAN/ground removal/association/tracking was executed"
    )


for name in (
    "dbscan",
    "remove_ground",
    "lidar_detect",
    "radar_detect",
    "associate_camera",
    "associate_radar",
    "linear_sum_assignment",
):
    setattr(perception, name, forbidden)
for name in ("lidar_detect", "radar_detect", "associate_camera", "associate_radar"):
    setattr(processor, name, forbidden)
tracking.Tracker.update = forbidden
tracking.linear_sum_assignment = forbidden
cfg = load_config("src/group8_fusion/config/fusion_ai.yaml")
cfg["model"]["weights"] = "/models/yolo11n.engine"
bag = Bag("/data/studentProject")
node = make_processor(
    cfg, bag.lidar_from("cam0"), bag.lidar_from("radar"), CameraDetector(cfg["model"])
)
rows = []
torch.cuda.reset_peak_memory_stats()
for i in range(a.frames):
    t, msg = bag.read(CAMERA, i)
    l = bag.nearest(LIDAR, t)
    r = bag.nearest(RADAR, t)
    result = node.process(
        image_array(msg),
        cloud_array(l[1], node.lidar_fields),
        cloud_array(r[1], node.radar_fields),
        t / 1e9,
    )
    assert result["metrics"]["pipeline"] == "neural"
    for track in result["tracks"]:
        assert np.isfinite(track.state).all() and np.isfinite(track.object.size).all()
    rows.append(
        {
            "frame": i,
            "ids": [track.id for track in result["tracks"]],
            "metrics": result["metrics"],
        }
    )
assert any(row["ids"] for row in rows), "No 3D tracks produced"
report = {
    "passed": True,
    "frames": a.frames,
    "forbidden_classical_calls": 0,
    "torch_peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
    "checkpoint_sha256": node.checkpoint_sha256,
    "observations": rows,
}
Path(a.output).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
bag.close()
