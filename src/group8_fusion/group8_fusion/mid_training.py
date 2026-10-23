"""Targets and diagnostics for a detector distilled from frozen LiDAR predictions.

The matching in this module is an evaluation calculation. It is never part of
the detector or its inference graph. Teacher agreement is not object accuracy.
"""

import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def preprocessing_contract(configuration):
    """Semantic feature settings that must agree between extraction and runtime."""
    camera = configuration["camera"]
    width, height = int(camera["width"]), int(camera["height"])
    intensity_scale = float(configuration["neural"]["lidar"].get("intensity_scale", 255.))
    if width < 1 or height < 1 or not math.isfinite(intensity_scale) or intensity_scale <= 0:
        raise ValueError("Camera dimensions and LiDAR intensity scale must be positive and finite")
    return dict(camera_width=width, camera_height=height, intensity_scale=intensity_scale,
                camera_lift_heights=[-1., 0., 1.], lidar_support_radius_cells=2)


def chronological_paths(root):
    """Return fixed non-overlapping training, selection and calibration segments."""
    root = Path(root)
    by_frame = {int(path.stem): path for path in (root / "studentProject").glob("*.npz")}
    required = set(range(235))
    if not required.issubset(by_frame):
        missing = sorted(required - set(by_frame))
        raise ValueError(f"Need bag 1 frames 0..234 for fixed chronological split; missing {missing[:10]}")
    development = sorted((root / "studentProject1").glob("*.npz"))
    if not development:
        raise ValueError("Second recording feature segment is empty")
    return dict(train=[by_frame[i] for i in range(164)],
                selection=[by_frame[i] for i in range(188, 211)],
                calibration=[by_frame[i] for i in range(211, 235)],
                development=development)


def teacher_targets(boxes, labels, scores, grid, threshold=.35, classes=10):
    """Keep finite, positive-sized teacher boxes whose centers are in the grid."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, 9)
    labels = np.asarray(labels, np.int64).reshape(-1)
    scores = np.asarray(scores, np.float32).reshape(-1)
    if not (len(boxes) == len(labels) == len(scores)):
        raise ValueError("Teacher boxes, labels and scores must have equal lengths")
    x = (boxes[:, 0] - grid["x_min"]) / grid["cell_size"]
    y = (boxes[:, 1] - grid["y_min"]) / grid["cell_size"]
    keep = (np.isfinite(boxes).all(1) & np.isfinite(scores)
            & (scores >= threshold) & (boxes[:, 3:6] > 0).all(1)
            & (labels >= 0) & (labels < classes)
            & (x >= 0) & (x < grid["width"])
            & (y >= 0) & (y < grid["height"]))
    return boxes[keep], labels[keep], scores[keep]


def gaussian_radius(height, width, overlap=.7):
    """CenterNet/CornerNet radius for a requested minimum box overlap."""
    a1, b1 = 1., height + width
    c1 = width * height * (1 - overlap) / (1 + overlap)
    r1 = (b1 + math.sqrt(max(0., b1 * b1 - 4 * a1 * c1))) / 2
    a2, b2 = 4., 2 * (height + width)
    c2 = (1 - overlap) * width * height
    r2 = (b2 + math.sqrt(max(0., b2 * b2 - 4 * a2 * c2))) / 2
    a3, b3 = 4 * overlap, -2 * overlap * (height + width)
    c3 = (overlap - 1) * width * height
    r3 = (b3 + math.sqrt(max(0., b3 * b3 - 4 * a3 * c3))) / 2
    return max(0, int(min(r1, r2, r3)))


def center_targets(boxes, labels, scores, grid, classes=10):
    """Build Gaussian heatmaps and one shared box target at each occupied cell.

    If two objects have the same center cell, only the higher confidence teacher
    supervises that cell. A single regression vector cannot represent both.
    """
    height, width = grid["height"], grid["width"]
    heatmap = np.zeros((classes, height, width), np.float32)
    regression = np.zeros((8, height, width), np.float32)
    mask = np.zeros((height, width), np.float32)
    weight = np.zeros((height, width), np.float32)
    occupied, collisions = set(), 0
    for index in np.argsort(-scores, kind="stable"):
        box, label = boxes[index], int(labels[index])
        gx = float((box[0] - grid["x_min"]) / grid["cell_size"])
        gy = float((box[1] - grid["y_min"]) / grid["cell_size"])
        ix, iy = int(gx), int(gy)
        if (iy, ix) in occupied:
            collisions += 1
            continue
        occupied.add((iy, ix))
        radius = min(12, gaussian_radius(box[4] / grid["cell_size"],
                                         box[3] / grid["cell_size"]))
        diameter = 2 * radius + 1
        axis = np.arange(-radius, radius + 1, dtype=np.float32)
        yy, xx = np.meshgrid(axis, axis, indexing="ij")
        gaussian = np.exp(-(xx * xx + yy * yy) / (2 * (diameter / 6) ** 2))
        left, right = min(ix, radius), min(width - ix - 1, radius)
        top, bottom = min(iy, radius), min(height - iy - 1, radius)
        area = heatmap[label, iy - top:iy + bottom + 1, ix - left:ix + right + 1]
        kernel = gaussian[radius - top:radius + bottom + 1,
                          radius - left:radius + right + 1]
        np.maximum(area, kernel, out=area)
        regression[:, iy, ix] = np.r_[gx - ix, gy - iy, box[2],
                                      np.log(box[3:6]), np.sin(box[6]), np.cos(box[6])]
        mask[iy, ix] = 1
        weight[iy, ix] = scores[index]
    return dict(heatmap=torch.from_numpy(heatmap),
                regression=torch.from_numpy(regression),
                mask=torch.from_numpy(mask), weight=torch.from_numpy(weight),
                collisions=collisions)


def detection_loss(prediction, target):
    """Gaussian focal center loss plus masked size, position and yaw regression."""
    logits = prediction["heatmap_logits"]
    labels = target["heatmap"]
    positive = labels.eq(1)
    negative_weight = (1 - labels).pow(4)
    probability = logits.sigmoid()
    positive_loss = F.logsigmoid(logits) * (1 - probability).pow(2) * positive
    negative_loss = (F.logsigmoid(-logits) * probability.pow(2)
                     * negative_weight * ~positive)
    focal = -(positive_loss.sum() + negative_loss.sum()) / positive.sum().clamp_min(1)
    reg = prediction["box_regression"]
    decoded = torch.cat((reg[:, :2].sigmoid(), reg[:, 2:]), 1)
    channel_weights = reg.new_tensor([1., 1., .5, 1., 1., 1., .5, .5])[None, :, None, None]
    weights = target["mask"] * target["weight"]
    error = F.smooth_l1_loss(decoded, target["regression"], reduction="none")
    box = (error * channel_weights * weights[:, None]).sum() / weights.sum().clamp_min(1)
    yaw_norm = (reg[:, 6:8].square().sum(1).clamp_min(1e-6).sqrt() - 1).square()
    yaw = (yaw_norm * weights).sum() / weights.sum().clamp_min(1)
    return focal + 2 * box + .02 * yaw, dict(heatmap=focal, box=box, yaw_norm=yaw)


def shift_camera(camera, available, dx, dy):
    """Translate camera BEV evidence with zero padding, without wrapping edges."""
    shifted = torch.roll(camera, shifts=(dy, dx), dims=(-2, -1))
    mask = torch.roll(available, shifts=(dy, dx), dims=(-2, -1))
    if dx > 0:
        shifted[..., :dx] = 0
        mask[..., :dx] = 0
    elif dx < 0:
        shifted[..., dx:] = 0
        mask[..., dx:] = 0
    if dy > 0:
        shifted[..., :dy, :] = 0
        mask[..., :dy, :] = 0
    elif dy < 0:
        shifted[..., dy:, :] = 0
        mask[..., dy:, :] = 0
    return shifted, mask


def agreement_metrics(predictions, teachers, threshold, match_distance=2.):
    """Score-ordered, same-class teacher matching within a fixed XY distance."""
    if len(predictions) != len(teachers):
        raise ValueError("Prediction and teacher frame counts must match")
    total_pred = total_teacher = matched = frames_with_predictions = invalid = 0
    center_xy, center_3d, size, yaw = [], [], [], []
    per_class = {str(i): dict(predictions=0, teachers=0, matches=0) for i in range(10)}
    for prediction, teacher in zip(predictions, teachers):
        boxes, scores, labels = (prediction[k] for k in ("boxes", "scores", "labels"))
        keep = scores >= threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        valid = (np.isfinite(boxes).all(1) & np.isfinite(scores)
                 & (boxes[:, 3:6] > 0).all(1))
        invalid += int((~valid).sum())
        boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
        tb, tl = teacher["boxes"], teacher["labels"]
        total_pred += len(boxes)
        total_teacher += len(tb)
        frames_with_predictions += int(len(boxes) > 0)
        used = set()
        for label in range(10):
            per_class[str(label)]["predictions"] += int((labels == label).sum())
            per_class[str(label)]["teachers"] += int((tl == label).sum())
        for index in np.argsort(-scores, kind="stable"):
            candidates = [j for j in range(len(tb)) if tl[j] == labels[index] and j not in used]
            if not candidates:
                continue
            distances = np.linalg.norm(tb[candidates, :2] - boxes[index, :2], axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] > match_distance:
                continue
            j = candidates[nearest]
            used.add(j)
            matched += 1
            per_class[str(int(labels[index]))]["matches"] += 1
            center_xy.append(float(distances[nearest]))
            center_3d.append(float(np.linalg.norm(tb[j, :3] - boxes[index, :3])))
            size.extend(np.abs(tb[j, 3:6] - boxes[index, 3:6]).tolist())
            difference = boxes[index, 6] - tb[j, 6]
            yaw.append(float(abs(np.arctan2(np.sin(difference), np.cos(difference)))))
    precision = matched / max(total_pred, 1)
    recall = matched / max(total_teacher, 1)
    return dict(frames=len(predictions), predictions=total_pred, teacher_objects=total_teacher,
                matched_teacher_objects=matched, teacher_center_precision=precision,
                teacher_center_recall=recall,
                teacher_center_f1=2 * precision * recall / max(precision + recall, 1e-12),
                frames_with_predictions=frames_with_predictions,
                prediction_frame_coverage=frames_with_predictions / max(len(predictions), 1),
                predictions_per_frame=total_pred / max(len(predictions), 1),
                invalid_predictions=invalid,
                matched_center_xy_mean_error_m=float(np.mean(center_xy)) if center_xy else None,
                matched_center_3d_mean_error_m=float(np.mean(center_3d)) if center_3d else None,
                matched_size_component_mae_m=float(np.mean(size)) if size else None,
                matched_yaw_mean_error_rad=float(np.mean(yaw)) if yaw else None,
                same_class_matching_distance_m=match_distance, per_class=per_class)
