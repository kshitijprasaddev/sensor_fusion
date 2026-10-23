"""Spatial middle fusion with an independent dense 3D detection head.

All inputs share a BEV grid. Camera, LiDAR and radar features are fused before
classification and box prediction. Neither forward nor decode accepts teacher
boxes, object candidates or proposal coordinates.

The eight regression channels are, in order, raw cell offsets x/y, centre z,
log dimensions x/y/z, sine yaw and cosine yaw. Decoding applies sigmoid to
offsets, clips z to [-4, 5] metres and exponentiates log dimensions clipped to
[-2, 3]. Grid rows run along y; columns run along x. Returned boxes have nine
columns [x, y, z, dx, dy, dz, yaw, vx, vy], with zero velocity.

This module is differentiable from spatial feature tensors to detections before
peak decoding. Fine-tuning the upstream sensor backbones is a separate concern.
"""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels):
    # Keep at least two channels per group, including on a 1 x 1 grid.
    return next(g for g in range(min(8, channels // 2), 0, -1) if channels % g == 0)


def _block(input_channels, output_channels, kernel=3):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel, padding=kernel // 2),
        nn.GroupNorm(_groups(output_channels), output_channels),
        nn.GELU(),
    )


class LocalWindowAttention(nn.Module):
    """Bound attention to at most 64 cells, including on non-square grids."""

    def __init__(self, dimension, window=8):
        super().__init__()
        self.window = min(8, max(1, int(window)))
        self.norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension, math.gcd(dimension, 4), batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, dimension * 2),
            nn.GELU(), nn.Linear(dimension * 2, dimension),
        )

    def forward(self, x, valid):
        batch, channels, height, width = x.shape
        window = self.window
        ph, pw = (-height) % window, (-width) % window
        x = F.pad(x, (0, pw, 0, ph))
        valid = F.pad(valid, (0, pw, 0, ph), value=False)
        padded_h, padded_w = height + ph, width + pw
        rows, columns = padded_h // window, padded_w // window
        tokens = x.reshape(batch, channels, rows, window, columns, window)
        tokens = tokens.permute(0, 2, 4, 3, 5, 1).reshape(-1, window * window, channels)
        mask = valid.reshape(batch, 1, rows, window, columns, window)
        mask = mask.permute(0, 2, 4, 3, 5, 1).reshape(-1, window * window)
        # Softmax over an entirely masked window is undefined. Expose one zero
        # sentinel in that window and discard every invalid output afterwards.
        safe_mask = mask.clone()
        safe_mask[~safe_mask.any(1), 0] = True
        normalized = self.norm(tokens)
        context, _ = self.attention(
            normalized, normalized, normalized,
            key_padding_mask=~safe_mask, need_weights=False,
        )
        tokens = tokens + context
        tokens = (tokens + self.feed_forward(tokens)) * mask[..., None]
        x = tokens.reshape(batch, rows, columns, window, window, channels)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, padded_h, padded_w)
        return x[:, :, :height, :width]


class BEVMiddleFusion(nn.Module):
    """Availability-aware feature fusion followed by dense independent heads.

    Availability has camera/LiDAR/radar channel order. Values are clipped to
    [0, 1]; non-finite, zero and negative availability mean absent. Missing
    input values are replaced before any convolution, so even NaN/Inf garbage
    in an absent modality cannot affect another modality. Completely absent
    cells emit zero gates/regression and a low heatmap logit of -20.
    """

    def __init__(self, camera_channels=192, lidar_channels=128, radar_channels=8,
                 dimension=32, classes=10):
        super().__init__()
        if dimension < 4 or classes < 1 or min(camera_channels, lidar_channels, radar_channels) < 1:
            raise ValueError("Feature dimensions must be positive and fusion dimension at least four")
        self.dimension = dimension
        self.classes = classes
        self.input_channels = (camera_channels, lidar_channels, radar_channels)
        self.camera_encoder = _block(camera_channels, dimension, kernel=1)
        self.lidar_encoder = _block(lidar_channels, dimension, kernel=1)
        self.radar_encoder = _block(radar_channels, dimension, kernel=1)
        self.gate = nn.Conv2d(dimension * 3 + 3, 3, 1)
        self.context = nn.Sequential(
            _block(dimension + 3, dimension), _block(dimension, dimension)
        )
        self.window_attention = LocalWindowAttention(dimension)
        self.heatmap_head = nn.Sequential(
            _block(dimension, dimension), nn.Conv2d(dimension, classes, 1)
        )
        self.box_head = nn.Sequential(
            _block(dimension, dimension), nn.Conv2d(dimension, 8, 1)
        )
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)
        nn.init.normal_(self.box_head[-1].weight, std=0.001)
        nn.init.zeros_(self.box_head[-1].bias)
        with torch.no_grad():
            self.box_head[-1].bias[7] = 1

    def forward(self, camera, lidar, radar, availability):
        inputs = (camera, lidar, radar)
        if camera.ndim != 4 or min(camera.shape[-2:]) < 1:
            raise ValueError("Expected nonempty [batch, channels, height, width] sensor features")
        batch, _, height, width = camera.shape
        if batch < 1 or availability.shape != (batch, 3, height, width):
            raise ValueError("Availability must have shape [batch, 3, height, width]")
        for x, channels in zip(inputs, self.input_channels):
            if x.shape != (batch, channels, height, width):
                raise ValueError("Sensor feature dimensions or BEV grids do not match")
            if x.device != camera.device or x.dtype != camera.dtype:
                raise ValueError("Sensor feature tensors must share a device and dtype")
        available = torch.nan_to_num(
            availability.to(device=camera.device, dtype=camera.dtype),
            nan=0, posinf=0, neginf=0,
        ).clamp(0, 1)
        valid = available > 0
        encoded = []
        for i, (x, encoder) in enumerate(zip(inputs, (
            self.camera_encoder, self.lidar_encoder, self.radar_encoder
        ))):
            clean = torch.where(valid[:, i:i + 1], x, torch.zeros_like(x))
            clean = torch.nan_to_num(clean, nan=0, posinf=0, neginf=0)
            encoded.append(encoder(clean) * valid[:, i:i + 1])
        logits = self.gate(torch.cat((*encoded, available), dim=1))
        logits = logits.masked_fill(~valid, -1e4)
        gates = logits.softmax(1) * available
        gates = gates / gates.sum(1, keepdim=True).clamp_min(torch.finfo(gates.dtype).tiny)
        fused = sum(feature * gates[:, i:i + 1] for i, feature in enumerate(encoded))
        any_available = valid.any(1, keepdim=True)
        fused = self.context(torch.cat((fused, available), dim=1)) * any_available
        fused = self.window_attention(fused, any_available)
        heatmap = self.heatmap_head(fused)
        regression = self.box_head(fused)
        return {
            "heatmap_logits": torch.where(any_available, heatmap, heatmap.new_full((), -20)),
            "box_regression": torch.where(any_available, regression, torch.zeros_like(regression)),
            "gates": gates,
        }


def decode(pred, grid, threshold=0.2, max_detections=64):
    """Decode one batch using local heatmap peaks; no object assignment solver.

    The grid must contain x_min, y_min, cell_size, width and height. Coordinates
    are cell lower boundaries plus sigmoid-regressed offsets, so a zero offset
    logit means the cell centre. Invalid regression vectors are discarded.
    """
    logits, regression = pred["heatmap_logits"], pred["box_regression"]
    if logits.ndim != 4 or logits.shape[0] != 1 or regression.shape != (1, 8, *logits.shape[-2:]):
        raise ValueError("Decode requires one batch with matching heatmap and eight-channel regression")
    height, width = logits.shape[-2:]
    if grid["height"] != height or grid["width"] != width:
        raise ValueError("Prediction grid dimensions differ from the configured BEV grid")
    if not all(math.isfinite(float(grid[k])) for k in ("x_min", "y_min", "cell_size")) or grid["cell_size"] <= 0:
        raise ValueError("Grid origin must be finite and cell size positive")
    if not 0 <= threshold <= 1 or max_detections < 0:
        raise ValueError("Threshold must be a probability and maximum detection count nonnegative")
    empty = {"boxes": np.empty((0, 9), np.float32),
             "scores": np.empty(0, np.float32), "labels": np.empty(0, np.int64)}
    if not max_detections:
        return empty
    with torch.no_grad():
        valid_logits = torch.isfinite(logits.detach())
        valid_regression = torch.isfinite(regression.detach()).all(1, keepdim=True)
        score = logits.detach().float().sigmoid()
        score = torch.nan_to_num(score, nan=0, posinf=0, neginf=0)
        peak = score == F.max_pool2d(score, 3, stride=1, padding=1)
        candidates = (
            peak & (score >= threshold) & valid_logits & valid_regression
        ).flatten().nonzero().flatten()
        if not len(candidates):
            return empty
        values = score.flatten()[candidates]
        order = values.argsort(descending=True, stable=True)[:max_detections]
        candidates, values = candidates[order], values[order]
        labels = candidates // (height * width)
        spatial = candidates % (height * width)
        rows, columns = spatial // width, spatial % width
        reg = regression.detach().float()[0, :, rows, columns].T
        finite = torch.isfinite(reg).all(1)
        if not finite.any():
            return empty
        reg, rows, columns = reg[finite], rows[finite], columns[finite]
        values, labels = values[finite], labels[finite]
        boxes = reg.new_zeros((len(reg), 9))
        offsets = reg[:, :2].sigmoid()
        boxes[:, 0] = grid["x_min"] + (columns + offsets[:, 0]) * grid["cell_size"]
        boxes[:, 1] = grid["y_min"] + (rows + offsets[:, 1]) * grid["cell_size"]
        boxes[:, 2] = reg[:, 2].clamp(-4, 5)
        boxes[:, 3:6] = reg[:, 3:6].clamp(-2, 3).exp()
        boxes[:, 6] = torch.atan2(reg[:, 6], reg[:, 7])
        return {"boxes": boxes.cpu().numpy(), "scores": values.cpu().numpy(),
                "labels": labels.cpu().numpy().astype(np.int64)}
