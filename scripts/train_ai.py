#!/usr/bin/env python3
"""Train fusion/GRU heads on frozen features with explicit sequence holdout.

Weak labels measure teacher agreement, never detection/tracking accuracy.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F
from group8_fusion.attention_fusion import QueryFusion

p = argparse.ArgumentParser()
p.add_argument("--features", default="/results/ai_features")
p.add_argument("--output", default="/models/query_fusion.pt")
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--report", default="/results/ai_training.json")
p.add_argument("--seed", type=int, default=8)
a = p.parse_args()
if a.epochs < 1:
    p.error("--epochs must be positive")
torch.manual_seed(a.seed)
np.random.seed(a.seed)
random.seed(a.seed)
torch.set_num_threads(2)
root = Path(a.features)
device = "cuda" if torch.cuda.is_available() else "cpu"
manifest = json.loads((root / "manifest.json").read_text())
files = sorted((root / "studentProject").glob("*.npz"))
if len(files) < 30:
    raise ValueError("Need at least 30 frames for disjoint train/validation segments")
# First 70% train, next 10% purge gap, final 20% validation. Bag 2 is never used
# in gradient updates or checkpoint selection. Features are frozen throughout.
cut = int(len(files) * 0.7)
val_start = int(len(files) * 0.8)


def load(paths):
    result = []
    for file in paths:
        with np.load(file, allow_pickle=False) as data:
            result.append(
                {k: torch.as_tensor(data[k], device=device) for k in data.files}
            )
    return result


train = load(files[:cut])
validation = load(files[val_start:])
positive = sum(int((r["target_lidar"] < len(r["boxes"])).sum()) for r in train)
if positive < 10:
    raise ValueError(
        "Insufficient positive weak associations; calibration/domain adaptation required"
    )
model = QueryFusion().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)


def sequence(records, training=False, drop=None):
    hidden = {}
    stats = {
        "queries": 0,
        "lidar_correct": 0,
        "radar_correct": 0,
        "lidar_positive": 0,
        "lidar_positive_correct": 0,
        "radar_positive": 0,
        "radar_positive_correct": 0,
        "predicted_lidar_nonnull": 0,
        "predicted_radar_nonnull": 0,
        "velocity_samples": 0,
        "velocity_absolute_error_sum": 0.0,
        "loss": 0.0,
        "frames": len(records),
    }
    block = []
    if training:
        optimizer.zero_grad()
    for index, r in enumerate(records):
        count = len(r["query"])
        if count:
            query = r["query"].float().clone()
            lidar = r["lidar"].float().clone()
            radar = r["radar"].float().clone()
            if drop == "camera_features":
                query[:, :256] = 0
            if drop == "lidar_features":
                lidar[:, :128] = 0
            if drop == "radar_features":
                radar[:] = 0
            fused, la, ra = model(query, lidar, radar, r["pair"].float())
            lt = r["target_lidar"].long()
            rt = r["target_radar"].long()
            loss = F.nll_loss(la.clamp_min(1e-8).log(), lt) + 0.5 * F.nll_loss(
                ra.clamp_min(1e-8).log(), rt
            )
            ids = list(map(int, r["ids"]))
            state = torch.stack(
                [
                    hidden.get(identifier, torch.zeros(model.dimension, device=device))
                    for identifier in ids
                ]
            )
            positions = torch.zeros((count, 3), device=device)
            valid = lt < len(r["boxes"])
            positions[valid] = r["boxes"][lt[valid], :3]
            velocity, state = model.predict_motion(
                fused, positions, r["previous"].float(), r["dt"].float(), state
            )
            hidden.update(zip(ids, state))
            motion_valid = valid & (r["dt"] > 0)
            if motion_valid.any():
                loss = loss + 0.1 * F.smooth_l1_loss(
                    velocity[motion_valid] / 20, r["target_velocity"][motion_valid] / 20
                )
                stats["velocity_absolute_error_sum"] += float(
                    (velocity[motion_valid] - r["target_velocity"][motion_valid])
                    .abs()
                    .sum()
                    .detach()
                )
                stats["velocity_samples"] += int(motion_valid.sum()) * 2
            stats["queries"] += count
            stats["loss"] += float(loss.detach())
            lc = la.argmax(1) == lt
            rc = ra.argmax(1) == rt
            rvalid = rt < len(r["radar"])
            stats["lidar_correct"] += int(lc.sum())
            stats["radar_correct"] += int(rc.sum())
            stats["lidar_positive"] += int(valid.sum())
            stats["lidar_positive_correct"] += int((lc & valid).sum())
            stats["radar_positive"] += int(rvalid.sum())
            stats["radar_positive_correct"] += int((rc & rvalid).sum())
            stats["predicted_lidar_nonnull"] += int(
                (la.argmax(1) < len(r["lidar"])).sum()
            )
            stats["predicted_radar_nonnull"] += int(
                (ra.argmax(1) < len(r["radar"])).sum()
            )
            if training:
                block.append(loss)
        if training and ((index + 1) % 8 == 0 or index == len(records) - 1):
            if block:
                torch.stack(block).mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()
                block = []
            hidden = {k: v.detach() for k, v in hidden.items()}
    q = max(stats["queries"], 1)
    stats["lidar_teacher_agreement"] = stats["lidar_correct"] / q
    stats["radar_teacher_agreement"] = stats["radar_correct"] / q
    stats["lidar_positive_teacher_recall"] = stats["lidar_positive_correct"] / max(
        stats["lidar_positive"], 1
    )
    stats["radar_positive_teacher_recall"] = stats["radar_positive_correct"] / max(
        stats["radar_positive"], 1
    )
    stats["velocity_weak_target_mae_mps"] = stats["velocity_absolute_error_sum"] / max(
        stats["velocity_samples"], 1
    )
    return stats


history = []
best_loss = float("inf")
best = None
best_epoch = 0
with torch.no_grad():
    initial = sequence(validation)
for epoch in range(a.epochs):
    model.train()
    tr = sequence(train, True)
    model.eval()
    with torch.no_grad():
        va = sequence(validation)
    score = va["loss"]
    history.append(
        {
            "epoch": epoch + 1,
            "train_loss": tr["loss"],
            "validation_loss": score,
            "validation_lidar_agreement": va["lidar_teacher_agreement"],
            "validation_radar_agreement": va["radar_teacher_agreement"],
        }
    )
    if score < best_loss:
        best_loss = score
        best = copy.deepcopy(model.state_dict())
        best_epoch = epoch + 1
    if epoch % 10 == 0:
        print(history[-1], flush=True)
model.load_state_dict(best)
model.eval()
with torch.no_grad():
    final_val = sequence(validation)
    # Evaluation is performed once after validation checkpoint selection.
    held_out = load(sorted((root / "studentProject1").glob("*.npz")))
    if not held_out:
        raise ValueError("Second recording required for held-out evaluation")
    test = sequence(held_out)
    ablations = {
        name: sequence(held_out, drop=name)
        for name in ("camera_features", "lidar_features", "radar_features")
    }
report = {
    "supervision": "weak_geometry",
    "metric_warning": "All agreement and velocity errors use pseudo-labels, NOT ground truth accuracy.",
    "seed": a.seed,
    "epochs": a.epochs,
    "best_epoch": best_epoch,
    "parameter_count": sum(v.numel() for v in model.parameters()),
    "split": {
        "train_bag": "studentProject",
        "train_frames": [0, cut - 1],
        "purged_frames": [cut, val_start - 1],
        "validation_frames": [val_start, len(files) - 1],
        "held_out_bag": "studentProject1",
        "held_out_frames": len(held_out),
    },
    "feature_manifest_sha256": hashlib.sha256(
        (root / "manifest.json").read_bytes()
    ).hexdigest(),
    "initial_validation": initial,
    "validation": final_val,
    "held_out": test,
    "feature_ablations": ablations,
    "history": history,
}
# Save CPU tensors and primitive provenance so weights_only=True is sufficient.
checkpoint = {
    "schema": "group8-query-fusion-v1",
    "supervision": "weak_geometry",
    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    "seed": a.seed,
    "best_epoch": best_epoch,
    "split": report["split"],
    "model_hashes": manifest["model_hashes"],
    "camera_intrinsics": manifest["configuration"]["camera"]["K"],
    "feature_manifest_sha256": report["feature_manifest_sha256"],
}
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
torch.save(checkpoint, a.output)
report["checkpoint_sha256"] = hashlib.sha256(Path(a.output).read_bytes()).hexdigest()
Path(a.report).write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {k: v for k, v in report.items() if k not in ("history", "feature_ablations")},
        indent=2,
    )
)
