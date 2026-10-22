#!/usr/bin/env python3
"""Cache frozen neural features and explicit weak targets from the two ROS bags."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from group8_fusion.bag import Bag, CAMERA, LIDAR, RADAR
from group8_fusion.geometry import image_array, cloud_array
from group8_fusion.processor import load_config, CameraDetector
from group8_fusion.neural import NeuralProcessor
from group8_fusion.attention_fusion import weak_targets

p = argparse.ArgumentParser()
p.add_argument("--output", default="/results/ai_features")
p.add_argument("--config", default="src/group8_fusion/config/fusion_ai.yaml")
p.add_argument("--limit", type=int, default=0)
p.add_argument("--bags", nargs="+", default=["studentProject", "studentProject1"])
a = p.parse_args()
if a.limit < 0:
    p.error("--limit must be nonnegative")
cfg = load_config(a.config)
cfg["model"]["weights"] = "/models/yolo11n.engine"
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
summary = {
    "supervision": "weak_geometry",
    "accuracy": "not ground truth",
    "bags": {},
    "configuration": cfg,
}
summary["model_hashes"] = {
    f.name: hashlib.sha256(f.read_bytes()).hexdigest()
    for f in Path("/models").iterdir()
    if f.name in ("voxelnext_nuscenes.pth", "sam2.1_t.pt", "yolo11n.engine")
}
processor = None
for name in a.bags:
    bag = Bag("/data/" + name)
    out = root / name
    out.mkdir(exist_ok=True)
    if list(out.glob("*.npz")):
        raise FileExistsError(
            f"Use a new output directory; features already exist: {out}"
        )
    if processor is None:
        processor = NeuralProcessor(
            cfg,
            bag.lidar_from("cam0"),
            bag.lidar_from("radar"),
            CameraDetector(cfg["model"]),
            training_features=True,
        )
    else:
        processor.camera_from_lidar = np.linalg.inv(bag.lidar_from("cam0"))
        processor.lidar_from_radar = bag.lidar_from("radar")
        processor.reset_tracking()
    rows = []
    previous = {}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for i in range(len(bag.index[CAMERA])):
        if a.limit and len(rows) >= a.limit:
            break
        timestamp, msg = bag.read(CAMERA, i)
        lidar = bag.nearest(LIDAR, timestamp)
        radar = bag.nearest(RADAR, timestamp)
        if lidar is None:
            continue
        image = image_array(msg)
        frame_start = time.perf_counter()
        record, visual, boxes = processor.extract(
            image,
            cloud_array(lidar[1], processor.lidar_fields),
            cloud_array(radar[1], processor.radar_fields) if radar else None,
            timestamp / 1e9,
        )
        lt, rt = weak_targets(record, visual)
        record["target_lidar"] = lt
        record["target_radar"] = rt
        record["frame"] = np.array(i)
        record["previous"] = np.zeros((len(lt), 3), np.float32)
        record["dt"] = np.zeros(len(lt), np.float32)
        record["target_velocity"] = np.zeros((len(lt), 2), np.float32)
        for j, target in enumerate(lt):
            if target >= len(record["boxes"]):
                continue
            identifier = int(record["ids"][j])
            pos = record["boxes"][target, :3]
            if identifier in previous:
                old_time, old = previous[identifier]
                dt = timestamp / 1e9 - old_time
                if 0 < dt < 0.5:
                    record["previous"][j] = old
                    record["dt"][j] = dt
                    record["target_velocity"][j] = np.clip(
                        (pos[:2] - old[:2]) / dt, -30, 30
                    )
            previous[identifier] = (timestamp / 1e9, pos.copy())
        np.savez_compressed(out / f"{i:05d}.npz", **record)
        rows.append(
            {
                "frame": i,
                "visual_tracks": len(visual),
                "lidar_detections": len(record["boxes"]),
                "radar_points": len(record["radar_xyz"]),
                "weak_lidar_matches": int((lt < len(record["boxes"])).sum()),
                "weak_radar_matches": int((rt < len(record["radar_xyz"])).sum()),
                "latency_ms": (time.perf_counter() - frame_start) * 1000,
            }
        )
        if i % 50 == 0:
            for v in visual:
                image[v["mask"]] = (
                    image[v["mask"]] * 0.65 + np.array([50, 200, 100]) * 0.35
                ).astype(np.uint8)
                x, y = v["bbox"][:2].astype(int)
                cv2.putText(
                    image,
                    f"SAM #{v['id']} {v['label']}",
                    (x, max(y - 10, 25)),
                    0,
                    0.8,
                    (50, 255, 100),
                    2,
                )
            cv2.imwrite(str(out / f"frame_{i:05d}.jpg"), cv2.resize(image, (960, 600)))
        if len(rows) % 10 == 0 or a.limit:
            print(name, len(rows), "frames;", rows[-1], flush=True)
    summary["bags"][name] = {
        "frames": len(rows),
        "wall_time_s": time.perf_counter() - started,
        "peak_torch_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "capacity_rejections": processor.visual_tracker.capacity_rejections,
        "per_frame": rows,
    }
    bag.close()
(root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
print(
    json.dumps(
        {
            k: {q: v for q, v in value.items() if q != "per_frame"}
            for k, value in summary["bags"].items()
        },
        indent=2,
    )
)
