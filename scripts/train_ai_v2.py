#!/usr/bin/env python3
"""Train and compare v2 using frozen features, with explicit development status."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F
from group8_fusion.robust_fusion import (
    RobustQueryFusion, load_fusion_model, radar_distribution,
)


def read_records(paths, device):
    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            records.append({k: torch.as_tensor(data[k], device=device) for k in data.files})
    return records


def run_sequence(model, records, optimizer=None, perturb=None):
    training = optimizer is not None
    model.train(training)
    hidden, block, calibration = {}, [], []
    stats = dict(queries=0, lidar_correct=0, radar_exact_correct=0,
                 radar_spatial_set_correct=0, lidar_nonnull=0, radar_nonnull=0,
                 lidar_accepted_correct=0, lidar_accepted=0,
                 radar_accepted_plausible=0, radar_accepted=0,
                 loss=0., velocity_error=0., velocity_components=0,
                 finite=True, frames=len(records))
    if training:
        optimizer.zero_grad()
    for frame, record in enumerate(records):
        r = record
        q, l, radar = (r[key].float().clone() for key in ("query", "lidar", "radar"))
        pair = r["pair"].float()
        lt = r["target_lidar"].long()
        rt = r["target_radar"].long()
        radar_xyz = r["radar_xyz"].float()
        if perturb == "camera_descriptor" or (training and random.random() < .35):
            q[:, :256] = 0
        if perturb == "lidar_descriptor" or (training and random.random() < .15):
            l[:, :128] = 0
        if perturb == "radar_attributes" or (training and random.random() < .15):
            radar[:, 3:5] = 0
        if perturb == "radar_missing" or (training and random.random() < .15):
            radar, radar_xyz = radar[:0], radar_xyz[:0]
            rt = torch.zeros_like(rt)
        if perturb == "lidar_missing":
            l, pair = l[:0], pair[:, :0]
            lt = torch.zeros_like(lt)
            rt = torch.full_like(rt, len(radar))
        if len(q):
            fused, la, ra = model(q, l, radar, pair)
            target_boxes = r["boxes"].float() if len(l) else r["boxes"][:0].float()
            distribution = radar_distribution(target_boxes, radar_xyz, lt)
            # Small label smoothing discourages infinite confidence in weak labels.
            onehot = F.one_hot(lt, len(l) + 1).float()
            ldist = onehot * .98 + .02 / (len(l) + 1)
            loss = -(ldist * la.clamp_min(1e-8).log()).sum(1).mean()
            loss = loss - .5 * (distribution * ra.clamp_min(1e-8).log()).sum(1).mean()
            ids = [int(v) for v in r["ids"]]
            state = torch.stack([hidden.get(v, fused.new_zeros(model.dimension)) for v in ids])
            position = fused.new_zeros((len(q), 3))
            valid = lt < len(l)
            position[valid] = target_boxes[lt[valid], :3]
            velocity, state = model.predict_motion(
                fused, position, r["previous"].float(), r["dt"].float(), state
            )
            hidden.update(zip(ids, state))
            motion_valid = valid & (r["dt"] > 0)
            if motion_valid.any():
                loss = loss + .1 * F.smooth_l1_loss(
                    velocity[motion_valid] / 20, r["target_velocity"][motion_valid] / 20
                )
                stats["velocity_error"] += float(
                    (velocity[motion_valid] - r["target_velocity"][motion_valid]).abs().sum().detach()
                )
                stats["velocity_components"] += int(motion_valid.sum()) * 2
            li, ri = la.argmax(1), ra.argmax(1)
            lc = li == lt
            plausible = distribution.gather(1, ri[:, None]).squeeze(1) > 0
            lac = (li < len(l)) & (la.max(1).values >= .65)
            rac = (ri < len(radar)) & (ra.max(1).values >= .35)
            stats["queries"] += len(q)
            stats["loss"] += float(loss.detach())
            stats["lidar_correct"] += int(lc.sum())
            stats["radar_exact_correct"] += int((ri == rt).sum())
            stats["radar_spatial_set_correct"] += int(plausible.sum())
            stats["lidar_nonnull"] += int((li < len(l)).sum())
            stats["radar_nonnull"] += int((ri < len(radar)).sum())
            stats["lidar_accepted"] += int(lac.sum())
            stats["lidar_accepted_correct"] += int((lac & lc).sum())
            stats["radar_accepted"] += int(rac.sum())
            stats["radar_accepted_plausible"] += int((rac & plausible).sum())
            stats["finite"] &= bool(torch.isfinite(fused).all() & torch.isfinite(velocity).all()
                                    & torch.isfinite(la).all() & torch.isfinite(ra).all())
            calibration.append((la.detach(), lt.detach(), ra.detach(), distribution.detach()))
            if training:
                block.append(loss)
        if training and ((frame + 1) % 8 == 0 or frame == len(records) - 1):
            if block:
                torch.stack(block).mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad()
                block = []
            hidden = {key: value.detach() for key, value in hidden.items()}
    n = max(stats["queries"], 1)
    for name in ("lidar_correct", "radar_exact_correct", "radar_spatial_set_correct"):
        stats[name + "_fraction"] = stats[name] / n
    stats["velocity_weak_mae_mps"] = stats["velocity_error"] / max(stats["velocity_components"], 1)
    stats["lidar_accepted_teacher_precision"] = stats["lidar_accepted_correct"] / max(stats["lidar_accepted"], 1)
    stats["radar_accepted_spatial_precision"] = stats["radar_accepted_plausible"] / max(stats["radar_accepted"], 1)
    return stats, calibration


def fit_temperatures(model, calibration):
    # Grid chosen in advance, optimized on a separate final segment of bag 1.
    result = {}
    for name, pindex, tindex in (("lidar", 0, 1), ("radar", 2, 3)):
        scores = []
        for temperature in (.5, .75, 1., 1.5, 2., 3., 5., 8.):
            losses = []
            for entry in calibration:
                probability, target = entry[pindex], entry[tindex]
                adjusted = RobustQueryFusion.temper(probability, temperature)
                if name == "lidar":
                    loss = F.nll_loss(adjusted.log(), target, reduction="sum")
                else:
                    loss = -(target * adjusted.log()).sum()
                losses.append((float(loss), len(target)))
            scores.append((sum(v[0] for v in losses) / max(sum(v[1] for v in losses), 1), temperature))
        best_loss, value = min(scores)
        getattr(model, name + "_temperature").fill_(value)
        result[name] = {"temperature": value, "calibration_nll": best_loss}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="/results/ai_features")
    parser.add_argument("--output", default="/models/query_fusion_v2.pt")
    parser.add_argument("--baseline", default="/models/query_fusion.pt")
    parser.add_argument("--report", default="/results/ai_training_v2.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=8)
    a = parser.parse_args()
    if a.epochs < 1:
        parser.error("--epochs must be positive")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(a.features)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    paths = sorted((root / "studentProject").glob("*.npz"))
    if len(paths) < 100:
        raise ValueError("Need at least 100 frames for disjoint training, purge, selection and calibration")
    cut, start, cal_start = int(.7 * len(paths)), int(.8 * len(paths)), int(.9 * len(paths))
    train = read_records(paths[:cut], device)
    selection = read_records(paths[start:cal_start], device)
    calibration = read_records(paths[cal_start:], device)
    if sum(int((r["target_lidar"] < len(r["boxes"])).sum()) for r in train) < 10:
        raise ValueError("Insufficient positive weak labels")
    model = RobustQueryFusion().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    history, best, best_loss, best_epoch = [], None, float("inf"), 0
    for epoch in range(a.epochs):
        tr, _ = run_sequence(model, train, optimizer)
        with torch.no_grad():
            va, _ = run_sequence(model, selection)
        history.append(dict(epoch=epoch + 1, train_loss=tr["loss"], selection_loss=va["loss"]))
        if va["loss"] < best_loss:
            best, best_loss, best_epoch = copy.deepcopy(model.state_dict()), va["loss"], epoch + 1
        if epoch % 10 == 0:
            print(history[-1], flush=True)
    model.load_state_dict(best)
    with torch.no_grad():
        _, values = run_sequence(model, calibration)
        temperatures = fit_temperatures(model, values)
        val, _ = run_sequence(model, selection + calibration)
    split = dict(train_bag="studentProject", train_frames=[0, cut - 1],
                 purged_frames=[cut, start - 1], selection_frames=[start, cal_start - 1],
                 temperature_calibration_frames=[cal_start, len(paths) - 1],
                 diagnostic_bag="studentProject1")
    checkpoint = dict(schema="group8-query-fusion-v2", supervision="weak_geometry",
                      state_dict={k: v.cpu().clone() for k, v in model.state_dict().items()},
                      seed=a.seed, best_epoch=best_epoch, split=split,
                      model_hashes=manifest["model_hashes"],
                      camera_intrinsics=manifest["configuration"]["camera"]["K"],
                      feature_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, a.output)
    # Bag 2 was already inspected during v1 development. It is now explicitly
    # a development diagnostic, never an untouched test of the new design.
    diagnostics = read_records(sorted((root / "studentProject1").glob("*.npz")), device)
    if not diagnostics:
        raise ValueError("Second bag required for paired development diagnostics")
    baseline = load_fusion_model(torch.load(a.baseline, map_location=device, weights_only=True)).to(device)
    with torch.no_grad():
        comparison = {"v1": run_sequence(baseline, diagnostics)[0], "v2": run_sequence(model, diagnostics)[0]}
        stress = {name: run_sequence(model, diagnostics, perturb=name)[0] for name in (
            "camera_descriptor", "lidar_descriptor", "radar_attributes", "radar_missing", "lidar_missing")}
    report = dict(schema=checkpoint["schema"], supervision="weak_geometry", seed=a.seed,
                  epochs=a.epochs, best_epoch=best_epoch, split=split, temperatures=temperatures,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  metric_warning="Teacher agreement and spatial plausibility only; no ground truth accuracy. Bag 2 is development data after prior inspection.",
                  checkpoint_sha256=hashlib.sha256(Path(a.output).read_bytes()).hexdigest(),
                  baseline_sha256=hashlib.sha256(Path(a.baseline).read_bytes()).hexdigest(),
                  feature_manifest_sha256=checkpoint["feature_manifest_sha256"],
                  validation=val, paired_development=comparison, stress=stress, history=history)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
