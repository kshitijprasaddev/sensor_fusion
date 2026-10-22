"""Project-specific learned object-query fusion, trained with disclosed weak labels.

Inspired by cross-attention fusion, not an implementation of BEVFusion/TransFusion.
No pretrained status is implied for these project-specific heads.
"""

import math
import numpy as np
import torch
from torch import nn
from .geometry import project, box_corners, iou, transform_points

CAMERA_CLASSES = ["car", "truck", "bus", "person", "motorcycle", "bicycle"]


def canonical(name):
    return {"pedestrian": "person", "construction_vehicle": "truck"}.get(name, name)


class QueryFusion(nn.Module):
    def __init__(self, dimension=64):
        super().__init__()
        self.dimension = dimension
        self.query = nn.Sequential(
            nn.LayerNorm(266), nn.Linear(266, dimension), nn.GELU()
        )
        self.lidar = nn.Sequential(
            nn.LayerNorm(152), nn.Linear(152, dimension), nn.GELU()
        )
        self.radar = nn.Sequential(
            nn.Linear(8, dimension), nn.GELU(), nn.Linear(dimension, dimension)
        )
        self.null_lidar = nn.Parameter(torch.zeros(1, dimension))
        self.null_radar = nn.Parameter(torch.zeros(1, dimension))
        self.lidar_bias = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, 1))
        self.radar_bias = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1))
        self.lidar_attention = nn.MultiheadAttention(dimension, 4, batch_first=True)
        self.radar_attention = nn.MultiheadAttention(dimension, 4, batch_first=True)
        self.output = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, dimension), nn.GELU()
        )
        self.motion = nn.GRUCell(dimension + 7, dimension)
        self.velocity = nn.Linear(dimension, 2)

    def forward(self, query, lidar, radar, pair_geometry):
        q = self.query(query)
        l = torch.cat((self.lidar(lidar), self.null_lidar), 0)
        r = torch.cat((self.radar(radar), self.null_radar), 0)
        bias = self.lidar_bias(pair_geometry).squeeze(-1)
        bias = torch.cat((bias, q.new_zeros(len(q), 1)), 1)
        context, weights_l = self.lidar_attention(
            q[None], l[None], l[None], attn_mask=bias
        )
        # Radar geometry is compared against soft LiDAR evidence, not a hard
        # Hungarian match. Attention includes a learned no-observation token.
        soft_position = weights_l[0, :, :-1] @ lidar[:, 128:131]
        if len(radar):
            uv = (
                radar[:, 5:7][None]
                - ((query[:, 256:258] + query[:, 258:260]) / 2)[:, None]
            )
            distances = torch.cdist(soft_position, radar[:, :3]).unsqueeze(-1)
            rbias = self.radar_bias(torch.cat((uv, distances), -1)).squeeze(-1)
        else:
            rbias = q.new_empty(len(q), 0)
        rbias = torch.cat((rbias, q.new_zeros(len(q), 1)), 1)
        fused = q + context[0]
        rcontext, weights_r = self.radar_attention(
            fused[None], r[None], r[None], attn_mask=rbias
        )
        return self.output(fused + rcontext[0]), weights_l[0], weights_r[0]

    def predict_motion(self, fused, position, previous, dt, hidden):
        x = torch.cat((fused, position / 65, previous / 65, dt.reshape(-1, 1)), 1)
        hidden = self.motion(x, hidden)
        return self.velocity(hidden) * 20, hidden


def projected_boxes(boxes, camera_from_lidar, K, width, height):
    result = np.full((len(boxes), 4), np.nan, np.float32)
    for i, box in enumerate(boxes):
        corners = box_corners(box[:3], box[3:6])
        rotation = np.array(
            [
                [np.cos(box[6]), -np.sin(box[6]), 0],
                [np.sin(box[6]), np.cos(box[6]), 0],
                [0, 0, 1],
            ]
        )
        corners = (corners - box[:3]) @ rotation.T + box[:3]
        uv, _, depth = project(corners, camera_from_lidar, K, width, height)
        if (depth > 0.1).all() and np.isfinite(uv).all():
            result[i] = np.r_[uv.min(0), uv.max(0)]
    return result


def pack_features(
    tracks, lidar, radar_raw, radar_transform, camera_transform, K, width, height
):
    boxes = lidar["boxes"]
    count = len(boxes)
    projected = projected_boxes(boxes, camera_transform, K, width, height)
    lclass = np.eye(10, dtype=np.float32)[lidar["labels"]]
    geometry = np.zeros((count, 9), np.float32)
    if count:
        geometry[:, : min(9, boxes.shape[1])] = boxes[:, :9]
        geometry[:, :3] /= 65
        geometry[:, 3:6] /= 10
        geometry[:, 6] /= math.pi
        geometry[:, 7:9] /= 20
    scale = np.array([width, height, width, height], np.float32)
    projection = np.nan_to_num(projected / scale, nan=-2, posinf=-2, neginf=-2)
    ltokens = np.c_[
        lidar["features"], geometry, lidar["scores"], lclass, projection
    ].astype(np.float32)
    q = np.empty((len(tracks), 266), np.float32)
    pair = np.zeros((len(tracks), count, 5), np.float32)
    for j, t in enumerate(tracks):
        classes = np.eye(6, dtype=np.float32)[CAMERA_CLASSES.index(t["label"])]
        q[j] = np.r_[t["pointer"], t["bbox"] / scale, classes]
        pair[j, :, :4] = np.clip(projection - t["bbox"] / scale, -4, 4)
        pair[j, :, 4] = [
            canonical(lidar["names"][label]) == t["label"] for label in lidar["labels"]
        ]
    radar_raw = np.asarray(radar_raw, np.float32).reshape(-1, 5)
    valid = np.isfinite(radar_raw[:, :3]).all(1)
    radar_raw = radar_raw[valid]
    xyz = transform_points(radar_raw[:, :3], radar_transform)
    keep = (
        (xyz[:, 0] > 2)
        & (xyz[:, 0] < 65)
        & (np.abs(xyz[:, 1]) < 20)
        & (np.abs(xyz[:, 2]) < 3)
    )
    xyz = xyz[keep]
    radar_raw = radar_raw[keep]
    # Bound attention cost using deterministic range order, without clustering.
    order = np.argsort(np.linalg.norm(xyz[:, :2], axis=1), kind="stable")[:128]
    xyz = xyz[order]
    radar_raw = radar_raw[order]
    uv, visible, _ = project(xyz, camera_transform, K, width, height)
    uv = np.nan_to_num(uv / np.array([width, height]), nan=-2, posinf=-2, neginf=-2)
    rtokens = np.c_[
        xyz / 65,
        np.nan_to_num(np.clip(radar_raw[:, 3] / 30, -3, 3), nan=0),
        np.nan_to_num(np.clip(radar_raw[:, 4] / 50, -4, 4), nan=0),
        uv,
        visible,
    ].astype(np.float32)
    return {
        "query": q,
        "lidar": ltokens,
        "radar": rtokens,
        "pair": pair,
        "boxes": boxes.astype(np.float32),
        "lidar_scores": lidar["scores"].astype(np.float32),
        "lidar_labels": lidar["labels"].astype(np.int64),
        "projected": projected,
        "radar_xyz": xyz.astype(np.float32),
        "radar_v": radar_raw[:, 3].astype(np.float32),
        "ids": np.array([t["id"] for t in tracks], np.int64),
        "labels": np.array(
            [CAMERA_CLASSES.index(t["label"]) for t in tracks], np.int64
        ),
    }


def weak_targets(record, tracks):
    """Projected-box supervision ONLY for training/evaluation; never runtime IDs.

    These are noisy pseudo-labels, not annotations. A teacher's mistakes can be
    inherited by the learned fusion head and must be measured with real labels.
    """
    targets = []
    radar_targets = []
    for t in tracks:
        scores = []
        for i, (box, label) in enumerate(
            zip(record["projected"], record["lidar_labels"])
        ):
            overlap = iou(t["bbox"], box) if np.isfinite(box).all() else 0
            compatible = (
                canonical(
                    [
                        "car",
                        "truck",
                        "construction_vehicle",
                        "bus",
                        "trailer",
                        "barrier",
                        "motorcycle",
                        "bicycle",
                        "pedestrian",
                        "traffic_cone",
                    ][label]
                )
                == t["label"]
            )
            scores.append(
                overlap * float(record["lidar_scores"][i]) if compatible else 0
            )
        target = (
            int(np.argmax(scores)) if scores and max(scores) > 0.015 else len(scores)
        )
        targets.append(target)
        rtarget = len(record["radar_xyz"])
        if target < len(scores) and len(record["radar_xyz"]):
            distances = np.linalg.norm(
                record["radar_xyz"][:, :2] - record["boxes"][target, :2], axis=1
            )
            if distances.min() < 3:
                rtarget = int(distances.argmin())
        radar_targets.append(rtarget)
    return np.array(targets, np.int64), np.array(radar_targets, np.int64)
