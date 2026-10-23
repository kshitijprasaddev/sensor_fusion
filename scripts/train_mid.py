#!/usr/bin/env python3
"""Train a middle fusion BEV detector with disclosed teacher distillation.

This freezes all published encoders. Bag 2 is development evidence, never an
independent test set. Calibration uses its own chronological segment of bag 1.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "group8_fusion"))
from group8_fusion.mid_model import BEVMiddleFusion, decode
from group8_fusion.mid_training import (
    agreement_metrics, center_targets, chronological_paths, detection_loss,
    preprocessing_contract, shift_camera, teacher_targets,
)


def read_records(paths, grid, teacher_threshold):
    records = []
    expected = {"camera": 192, "lidar": 128, "radar": 8, "availability": 3}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            record = {key: torch.from_numpy(data[key].copy()) for key in expected}
            for key, channels in expected.items():
                value = record[key]
                shape = (channels, grid["height"], grid["width"])
                if tuple(value.shape) != shape or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid {key} feature map in {path}: expected finite {shape}")
            boxes, labels, scores = teacher_targets(
                data["teacher_boxes"], data["teacher_labels"], data["teacher_scores"],
                grid, threshold=teacher_threshold,
            )
            record["teacher"] = dict(boxes=boxes, labels=labels, scores=scores)
            record["target"] = center_targets(boxes, labels, scores, grid)
            occupied = record["target"]["mask"].bool()
            support = (record["availability"] > 0).any(0)
            record["teacher_support"] = dict(
                supervised_centers=int(occupied.sum()),
                centers_without_sensor_support=int((occupied & ~support).sum()),
            )
            record["path"] = str(path)
            records.append(record)
    if not records:
        raise ValueError("Feature segment is empty")
    return records


def batch_inputs(records, device, training=False, perturb=None, lidar_only=False,
                 modality_dropout=.15, camera_jitter=True):
    inputs = {key: torch.stack([record[key] for record in records]).to(device).float()
              for key in ("camera", "lidar", "radar", "availability")}
    unavailable = torch.zeros((len(records), 3), device=device, dtype=torch.bool)
    if training:
        unavailable = torch.rand((len(records), 3), device=device) < modality_dropout
        # Explicit empty examples train a meaningful no-observation response.
        unavailable[torch.rand(len(records), device=device) < .03] = True
        if camera_jitter:
            for index in range(len(records)):
                if random.random() < .35:
                    dx, dy = random.randint(-1, 1), random.randint(-1, 1)
                    camera, available = shift_camera(inputs["camera"][index],
                                                     inputs["availability"][index, 0], dx, dy)
                    inputs["camera"][index] = camera
                    inputs["availability"][index, 0] = available
    for index, name in enumerate(("camera", "lidar", "radar")):
        if perturb == name + "_missing" or perturb == "all_missing" or (lidar_only and name != "lidar"):
            unavailable[:, index] = True
        inputs[name][unavailable[:, index]] = 0
        inputs["availability"][:, index][unavailable[:, index]] = 0
    if perturb in ("camera_jitter_1cell", "camera_jitter_2cell"):
        amount = 1 if perturb == "camera_jitter_1cell" else 2
        inputs["camera"], inputs["availability"][:, 0] = shift_camera(
            inputs["camera"], inputs["availability"][:, 0], amount, -amount)
    return inputs


def run_training_epoch(model, records, optimizer, device, args, lidar_only=False):
    order = list(range(len(records)))
    random.shuffle(order)
    accumulated = dict(total=0., heatmap=0., box=0., yaw_norm=0.)
    model.train()
    for start in range(0, len(order), args.batch_size):
        rows = [records[index] for index in order[start:start + args.batch_size]]
        inputs = batch_inputs(rows, device, True, lidar_only=lidar_only,
                              modality_dropout=args.modality_dropout,
                              camera_jitter=not args.no_camera_jitter)
        target = {key: torch.stack([record["target"][key] for record in rows]).to(device)
                  for key in ("heatmap", "regression", "mask", "weight")}
        missing = inputs["availability"].flatten(1).sum(1) == 0
        for key in target:
            target[key][missing] = 0
        optimizer.zero_grad(set_to_none=True)
        prediction = model(**inputs)
        loss, components = detection_loss(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        accumulated["total"] += float(loss.detach()) * len(rows)
        for key, value in components.items():
            accumulated[key] += float(value.detach()) * len(rows)
    return {key: value / len(records) for key, value in accumulated.items()}


@torch.no_grad()
def evaluate(model, records, grid, device, threshold=.05, perturb=None, lidar_only=False):
    model.eval()
    predictions, gate_sum = [], np.zeros(3, np.float64)
    total_loss, all_finite, elapsed = 0., True, []
    for record in records:
        inputs = batch_inputs([record], device, perturb=perturb, lidar_only=lidar_only)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = model(**inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - started) * 1000)
        all_finite &= all(bool(torch.isfinite(value).all()) for value in result.values())
        if not all_finite:
            raise FloatingPointError("Non-finite detector output")
        target = {key: record["target"][key][None].to(device)
                  for key in ("heatmap", "regression", "mask", "weight")}
        loss, _ = detection_loss(result, target)
        total_loss += float(loss)
        predictions.append(decode(result, grid, threshold=threshold, max_detections=64))
        gate_sum += result["gates"].float().mean((0, 2, 3)).cpu().numpy()
    return predictions, dict(loss=total_loss / len(records), all_outputs_finite=all_finite,
                             mean_sensor_gates=dict(zip(("camera", "lidar", "radar"),
                                                       (gate_sum / len(records)).tolist())),
                             head_latency_p50_ms=float(np.median(elapsed)),
                             head_latency_p95_ms=float(np.percentile(elapsed, 95)),
                             latency_warning="Cached-feature head only, excludes all frozen encoders")


def score(predictions, records, threshold, diagnostics=None):
    result = agreement_metrics(predictions, [record["teacher"] for record in records], threshold)
    if diagnostics:
        result.update(diagnostics)
    return result


def train_one(args, segments, grid, manifest, feature_hash, device, lidar_only=False):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = BEVMiddleFusion(camera_channels=192, lidar_channels=128,
                            radar_channels=8, dimension=32, classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
    initial_predictions, initial_diagnostics = evaluate(
        model, segments["selection"], grid, device, lidar_only=lidar_only)
    initial = score(initial_predictions, segments["selection"], .2, initial_diagnostics)
    best, best_loss, selected_epoch, history = None, float("inf"), 0, []
    for epoch in range(args.epochs):
        train = run_training_epoch(model, segments["train"], optimizer, device, args, lidar_only)
        predictions, diagnostics = evaluate(model, segments["selection"], grid, device,
                                            lidar_only=lidar_only)
        selection = score(predictions, segments["selection"], .2, diagnostics)
        row = dict(epoch=epoch + 1, learning_rate=optimizer.param_groups[0]["lr"],
                   train=train, selection_loss=selection["loss"],
                   selection_teacher_f1_at_020=selection["teacher_center_f1"])
        history.append(row)
        if selection["loss"] < best_loss:
            best_loss, selected_epoch = selection["loss"], epoch + 1
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            print(json.dumps(dict(experiment="lidar_only" if lidar_only else "middle_fusion", **row)), flush=True)
    if best is None:
        raise RuntimeError("No finite model checkpoint selected")
    model.load_state_dict(best)
    calibration_predictions, calibration_diagnostics = evaluate(
        model, segments["calibration"], grid, device, lidar_only=lidar_only)
    thresholds = (.10, .15, .20, .25, .30, .35, .40, .45, .50, .60, .70, .80, .90)
    calibration_table = [dict(threshold=value, **score(calibration_predictions,
                                                      segments["calibration"], value))
                         for value in thresholds]
    # Fixed grid; maximize teacher F1 and break ties conservatively at higher scores.
    selected = max(calibration_table, key=lambda item: (item["teacher_center_f1"], item["threshold"]))
    threshold = selected["threshold"]
    metrics = {}
    for name in ("selection", "calibration", "development"):
        if name == "calibration":
            predictions, diagnostics = calibration_predictions, calibration_diagnostics
        else:
            predictions, diagnostics = evaluate(model, segments[name], grid, device,
                                                lidar_only=lidar_only)
        metrics[name] = score(predictions, segments[name], threshold, diagnostics)
    perturbations = {}
    for perturb in ("camera_missing", "lidar_missing", "radar_missing", "all_missing",
                    "camera_jitter_1cell", "camera_jitter_2cell"):
        predictions, diagnostics = evaluate(model, segments["development"], grid, device,
                                            perturb=perturb, lidar_only=lidar_only)
        perturbations[perturb] = score(predictions, segments["development"], threshold, diagnostics)
    report = dict(supervision="teacher_distillation", seed=args.seed, epochs=args.epochs,
                  selected_epoch=selected_epoch, selected_score_threshold=threshold,
                  parameter_count=sum(parameter.numel() for parameter in model.parameters()),
                  inputs="lidar_only" if lidar_only else "camera_lidar_radar_bev_features",
                  initial_selection=initial, metrics=metrics, robustness=perturbations,
                  threshold_calibration=calibration_table, history=history,
                  feature_manifest_sha256=feature_hash,
                  target_center_cell_collisions={name: sum(record["target"]["collisions"] for record in rows)
                                                 for name, rows in segments.items()},
                  target_center_sensor_support={
                      name: {key: sum(record["teacher_support"][key] for record in rows)
                             for key in ("supervised_centers", "centers_without_sensor_support")}
                      for name, rows in segments.items()
                  },
                  training=dict(optimizer="AdamW", learning_rate=args.learning_rate,
                                weight_decay=.01, batch_size=args.batch_size,
                                modality_dropout_probability=args.modality_dropout,
                                all_missing_probability=.03,
                                camera_jitter="35% probability, integer shift +/- 1 BEV cell" if not args.no_camera_jitter else "disabled",
                                teacher_score_minimum=args.teacher_threshold,
                                losses="Gaussian center focal + 2 * weighted smooth-L1 box + 0.02 * yaw norm",
                                encoders="All frozen; only middle fusion and detection layers trained"))
    checkpoint = dict(schema="group8-middle-fusion-v1", supervision="teacher_distillation",
                      state_dict=best, grid=grid, seed=args.seed, selected_epoch=selected_epoch,
                      score_threshold=threshold, model_hashes=manifest["model_hashes"],
                      camera_intrinsics=manifest["configuration"]["camera"]["K"],
                      feature_contract=preprocessing_contract(manifest["configuration"]),
                      feature_manifest_sha256=feature_hash,
                      experiment="lidar_only" if lidar_only else "middle_fusion",
                      architecture=dict(camera_channels=192, lidar_channels=128,
                                        radar_channels=8, dimension=32, classes=10))
    return checkpoint, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="/results/middle_features")
    parser.add_argument("--output", default="/models/middle_fusion.pt")
    parser.add_argument("--report", default="/results/middle_training.json")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--teacher-threshold", type=float, default=.35)
    parser.add_argument("--modality-dropout", type=float, default=.15)
    parser.add_argument("--no-camera-jitter", action="store_true")
    parser.add_argument("--lidar-baseline", action="store_true",
                        help="Also train an identical head using LiDAR only for controlled comparison")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch size and learning rate must be positive")
    if not 0 <= args.modality_dropout < 1 or not 0 <= args.teacher_threshold <= 1:
        parser.error("dropout and teacher threshold must be in [0, 1]")
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.features)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    grid = manifest["grid"]
    feature_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    paths = chronological_paths(root)
    print(json.dumps(dict(device=str(device), loading_frames={key: len(value) for key, value in paths.items()})), flush=True)
    segments = {name: read_records(files, grid, args.teacher_threshold) for name, files in paths.items()}
    if sum(len(record["teacher"]["boxes"]) for record in segments["train"]) < 10:
        raise ValueError("Need at least 10 in-grid, confidence-filtered training teacher boxes")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    checkpoint, experiment = train_one(args, segments, grid, manifest, feature_hash, device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    report = dict(schema="group8-middle-training-report-v1",
                  metric_warning="All matching and error metrics compare against frozen VoxelNeXt teacher predictions. They are NOT ground-truth detection accuracy, mAP, HOTA or evidence of state-of-the-art performance.",
                  evaluation_status="Both available recordings have already informed project development. Bag 2 is a development diagnostic, not an untouched test set.",
                  learning_scope="Frozen camera/LiDAR encoders with a jointly trained feature fusion and box detection head; not raw-sensor end-to-end training.",
                  split=dict(train_bag="studentProject", train_frames=[0, 163],
                             purged_frames=[164, 187], checkpoint_selection_frames=[188, 210],
                             threshold_calibration_frames=[211, 234],
                             development_bag="studentProject1", development_frames=len(segments["development"]),
                             independent_test_set_available=False),
                  grid=grid, model_hashes=manifest["model_hashes"],
                  feature_contract=preprocessing_contract(manifest["configuration"]),
                  feature_manifest_sha256=feature_hash, middle_fusion=experiment,
                  checkpoint_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    if args.lidar_baseline:
        baseline_checkpoint, baseline = train_one(args, segments, grid, manifest, feature_hash, device, lidar_only=True)
        baseline_path = output.with_name(output.stem + "_lidar_only" + output.suffix)
        torch.save(baseline_checkpoint, baseline_path)
        baseline["checkpoint_sha256"] = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        report["lidar_only_control"] = baseline
        report["controlled_comparison_warning"] = "Identical head and split, with independently calibrated thresholds; teacher is LiDAR-only, which favors the LiDAR control. Cross-sensor superiority cannot be inferred from this teacher diagnostic."
    report["elapsed_seconds"] = time.perf_counter() - started
    report["peak_torch_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else None
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("middle_fusion", "lidar_only_control")}, indent=2), flush=True)
    print(json.dumps(dict(selected_epoch=experiment["selected_epoch"],
                          score_threshold=experiment["selected_score_threshold"],
                          development=experiment["metrics"]["development"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
