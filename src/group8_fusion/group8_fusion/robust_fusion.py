"""QueryFusion v2: gated descriptors, contextual radar sets and soft supervision.

This is a small project-specific research head, not a reproduction of a named
published architecture. Geometry remains necessary and weak labels remain noisy.
"""

import torch
from torch import nn
from .attention_fusion import QueryFusion


class RobustQueryFusion(QueryFusion):
    def __init__(self, dimension=64):
        super().__init__(dimension)
        # Normalize descriptors independently: their statistics should not rescale
        # coordinates/classes. Whole-descriptor dropout is applied in training.
        del self.query, self.lidar
        self.camera_descriptor = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, dimension), nn.GELU()
        )
        self.camera_geometry = nn.Sequential(nn.Linear(10, dimension), nn.GELU())
        self.camera_gate = nn.Linear(dimension, 1)
        nn.init.constant_(self.camera_gate.bias, -2)
        self.lidar_descriptor = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, dimension), nn.GELU()
        )
        self.lidar_geometry = nn.Sequential(nn.Linear(24, dimension), nn.GELU())
        self.radar_context = nn.TransformerEncoderLayer(
            dimension, 4, dim_feedforward=128, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.radar_bias = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, 1))
        self.register_buffer("lidar_temperature", torch.ones(()))
        self.register_buffer("radar_temperature", torch.ones(()))

    @staticmethod
    def temper(probability, temperature):
        return (probability.clamp_min(1e-8).log() / temperature).softmax(-1)

    def forward(self, query, lidar, radar, pair_geometry):
        descriptor = self.camera_descriptor(query[:, :256])
        gate = self.camera_gate(descriptor).sigmoid()
        q = self.camera_geometry(query[:, 256:]) + gate * descriptor
        l = self.lidar_geometry(lidar[:, 128:]) + self.lidar_descriptor(lidar[:, :128])
        l = torch.cat((l, self.null_lidar), 0)
        # The null token keeps empty radar sets valid. Self-attention lets each
        # return encode surrounding evidence before cross-modal query attention.
        r = torch.cat((self.radar(radar), self.null_radar), 0)
        r = self.radar_context(r[None])[0]
        bias = self.lidar_bias(pair_geometry).squeeze(-1)
        bias = torch.cat((bias, q.new_zeros(len(q), 1)), 1)
        context, la = self.lidar_attention(q[None], l[None], l[None], attn_mask=bias)
        la = la[0]
        soft_position = la[:, :-1] @ lidar[:, 128:131]
        uv = radar[:, 5:7][None] - (
            (query[:, 256:258] + query[:, 258:260]) / 2
        )[:, None]
        xyz = radar[None, :, :3] - soft_position[:, None]
        rbias = self.radar_bias(torch.cat((xyz, uv), -1)).squeeze(-1)
        rbias = torch.cat((rbias, q.new_zeros(len(q), 1)), 1)
        fused = q + context[0]
        rcontext, ra = self.radar_attention(
            fused[None], r[None], r[None], attn_mask=rbias
        )
        return (
            self.output(fused + rcontext[0]),
            self.temper(la, self.lidar_temperature),
            self.temper(ra[0], self.radar_temperature),
        )


def radar_distribution(boxes, radar_xyz, lidar_target, radius=3.0, sigma=1.5):
    """Weak spatial distribution, not physical return-to-object ground truth.

    Multiple points within radius can be plausible. Empty sets and missing
    LiDAR teachers supervise the explicit null token rather than a forced match.
    """
    count, points = len(lidar_target), len(radar_xyz)
    distribution = boxes.new_zeros((count, points + 1))
    for j, index in enumerate(lidar_target):
        if int(index) < len(boxes) and points:
            distance = torch.linalg.vector_norm(
                radar_xyz[:, :2] - boxes[index, :2], dim=1
            )
            weight = torch.exp(-0.5 * (distance / sigma).square())
            weight = weight * (distance < radius)
            if weight.sum() > 0:
                distribution[j, :-1] = weight / weight.sum()
                continue
        distribution[j, -1] = 1
    return distribution


def load_fusion_model(checkpoint):
    schema = checkpoint.get("schema")
    if checkpoint.get("supervision") != "weak_geometry":
        raise ValueError("Unsupported fusion supervision or missing provenance")
    if schema == "group8-query-fusion-v1":
        model = QueryFusion()
    elif schema == "group8-query-fusion-v2":
        model = RobustQueryFusion()
    else:
        raise ValueError("Unsupported fusion checkpoint schema")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model
