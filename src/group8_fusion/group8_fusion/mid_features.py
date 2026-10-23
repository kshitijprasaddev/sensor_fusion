"""Current-image spatial encoders and metric BEV alignment.

The default job freezes published encoders. Tensor operations also permit a
separate backward integration check; that does not imply joint fine-tuning.
"""
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .neural_models import VoxelNeXtDetector

GRID = dict(x_min=0., y_min=-24., cell_size=.75, width=64, height=64)
NAMES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
         'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']


def centers(grid, device):
    y, x = torch.meshgrid(torch.arange(grid['height'], device=device),
                          torch.arange(grid['width'], device=device), indexing='ij')
    return torch.stack((grid['x_min'] + (x + .5) * grid['cell_size'],
                        grid['y_min'] + (y + .5) * grid['cell_size']), -1).float()


class CameraSpatialEncoder(nn.Module):
    """YOLO11 FPN features before Detect; no proposals or decoded boxes needed."""
    def __init__(self, weights='/models/yolo11n.pt', device='cuda'):
        super().__init__()
        from ultralytics import YOLO
        self.network = YOLO(weights).model.to(device).eval()
        self.network.requires_grad_(False)
        self.device = device

    def tensor(self, image):
        rgb = cv2.cvtColor(cv2.resize(image, (640, 400)), cv2.COLOR_BGR2RGB)
        value = torch.as_tensor(rgb.copy(), device=self.device).permute(2, 0, 1)[None].float() / 255
        return F.pad(value, (0, 0, 120, 120), value=114/255)

    def forward(self, image_tensor):
        outputs, value = [], image_tensor
        for layer in self.network.model:
            if layer.i > 19:
                break
            source = layer.f
            value = value if source == -1 else (
                outputs[source] if isinstance(source, int) else
                [value if index == -1 else outputs[index] for index in source]
            )
            value = layer(value)
            outputs.append(value)
        if outputs[16].shape[1] != 64 or outputs[19].shape[1] != 128:
            raise RuntimeError('Unexpected YOLO11n feature schema; use the pinned published checkpoint')
        return torch.cat((outputs[16], F.interpolate(outputs[19], size=outputs[16].shape[-2:], mode='bilinear', align_corners=False)), 1)


class LidarSpatialEncoder(nn.Module):
    """VoxelNeXt representation before its detection head, optionally with teacher labels."""
    def __init__(self, configuration, detector=None):
        super().__init__()
        self.detector = detector if detector is not None else VoxelNeXtDetector(dict(configuration, deterministic=True))
        self.network = self.detector.model
        self.network.requires_grad_(False)
        self.network.eval()
        dataset = getattr(self.detector, 'dataset', None)
        self.point_cloud_range = np.asarray(
            getattr(dataset, 'point_cloud_range', [-54., -54., -5., 54., 54., 3.]),
            dtype=np.float32,
        )

    def _empty(self, teacher):
        # spconv cannot process an empty active-site set. Preserve the same
        # device and sparse schema without invoking voxelization or CUDA ops.
        device = next(self.network.parameters()).device
        result = dict(features=torch.empty((0, 128), device=device),
                      indices=torch.empty((0, 3), device=device, dtype=torch.int32),
                      shape=(180, 180))
        if teacher:
            result.update(teacher_boxes=np.empty((0, 9), np.float32),
                          teacher_labels=np.empty(0, np.int64),
                          teacher_scores=np.empty(0, np.float32))
        return result

    def forward(self, raw, teacher=False):
        raw = np.asarray(raw, np.float32)
        if raw.ndim != 2 or raw.shape[1] != 4:
            raise ValueError('Spatial LiDAR encoder requires XYZ and recorded intensity')
        keep = np.isfinite(raw).all(1)
        keep &= (raw[:, :3] >= self.point_cloud_range[:3]).all(1)
        keep &= (raw[:, :3] < self.point_cloud_range[3:]).all(1)
        points = raw[keep].copy()
        if not len(points):
            return self._empty(teacher)
        points[:, 3] = np.clip(points[:, 3] / self.detector.intensity_scale, 0, 1)
        points = np.c_[points, np.zeros(len(points), np.float32)]
        data = self.detector.dataset.prepare_data(dict(points=points, frame_id=0))
        if not len(data['voxels']) or not len(data['voxel_coords']):
            return self._empty(teacher)
        from pcdet.models import load_data_to_gpu
        batch = self.detector.dataset.collate_batch([data])
        load_data_to_gpu(batch)
        for module in self.network.module_list:
            if module is self.network.dense_head:
                break
            batch = module(batch)
        sparse = batch['encoded_spconv_tensor']
        result = dict(features=sparse.features, indices=sparse.indices,
                      shape=tuple(map(int, sparse.spatial_shape)))
        if teacher:
            # This branch is for pseudo-label extraction only. Runtime never
            # calls it, and the guard replaces this head with a failing stub.
            self.network.dense_head(batch)
            prediction = batch['final_box_dicts'][0]
            keep = prediction['pred_scores'] >= .35
            result.update(teacher_boxes=prediction['pred_boxes'][keep].detach().cpu().numpy(),
                          teacher_labels=prediction['pred_labels'][keep].detach().cpu().numpy().astype(np.int64) - 1,
                          teacher_scores=prediction['pred_scores'][keep].detach().cpu().numpy())
            self.network.dense_head.forward_ret_dict.clear()
            self.detector.sample = None
        return result


def lidar_bev(sparse, grid):
    values, indices = sparse['features'], sparse['indices']
    height, width = sparse['shape']
    dense = values.new_zeros((1, 128, height, width))
    valid = values.new_zeros((1, 1, height, width))
    dense[0, :, indices[:, 1].long(), indices[:, 2].long()] = values.T
    valid[0, 0, indices[:, 1].long(), indices[:, 2].long()] = 1
    xy = centers(grid, values.device)
    index = (xy + 54.) / .6
    coordinates = torch.stack(((index[..., 0] + .5) * 2 / width - 1,
                               (index[..., 1] + .5) * 2 / height - 1), -1)[None]
    features = F.grid_sample(dense, coordinates, align_corners=False)[0]
    support = F.grid_sample(valid, coordinates, align_corners=False)
    # Returns lie on object surfaces, while a box centre may be in the empty
    # interior. Let the context layers predict within two neighbouring cells.
    # This is local feature support, not a claim of measured point occupancy;
    # the feature values themselves are neither dilated nor fabricated.
    support = F.max_pool2d(support, kernel_size=5, stride=1, padding=2)[0]
    return features, support


def camera_bev(features, grid, camera_from_lidar, intrinsics, width=1920, height=1200):
    xy = centers(grid, features.device)
    matrix = torch.as_tensor(camera_from_lidar, dtype=features.dtype, device=features.device)
    K = torch.as_tensor(intrinsics, dtype=features.dtype, device=features.device).reshape(3, 3)
    samples, masks = [], []
    for z in (-1., 0., 1.):
        xyz = torch.cat((xy, xy.new_full((*xy.shape[:2], 1), z)), -1)
        camera = xyz @ matrix[:3, :3].T + matrix[:3, 3]
        pixels = camera @ K.T
        uv = pixels[..., :2] / pixels[..., 2:].clamp_min(.01)
        valid = (camera[..., 2] > .1) & (uv[..., 0] >= 0) & (uv[..., 0] < width) & (uv[..., 1] >= 0) & (uv[..., 1] < height)
        # Original 1920x1200 -> 640x400, centered inside the 640x640 input.
        padded = torch.stack((uv[..., 0] * (640 / width), uv[..., 1] * (400 / height) + 120), -1)
        coordinates = padded * (2 / 640) - 1
        sampled = F.grid_sample(features, coordinates[None], align_corners=False)[0]
        samples.append(sampled * valid[None])
        masks.append(valid.float()[None])
    support = torch.stack(masks).sum(0)
    return torch.stack(samples).sum(0) / support.clamp_min(1), support / 3


def radar_bev(raw, lidar_from_radar, grid, device='cuda'):
    raw = torch.as_tensor(raw, dtype=torch.float32, device=device).reshape(-1, 5)
    matrix = torch.as_tensor(lidar_from_radar, dtype=torch.float32, device=device)
    xyz = raw[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]
    xy_index = torch.floor((xyz[:, :2] - xyz.new_tensor([grid['x_min'], grid['y_min']])) / grid['cell_size']).long()
    keep = torch.isfinite(xyz).all(1) & (xyz[:, 2].abs() < 3) & (xy_index[:, 0] >= 0) & (xy_index[:, 0] < grid['width']) & (xy_index[:, 1] >= 0) & (xy_index[:, 1] < grid['height'])
    xyz, raw, xy_index = xyz[keep], raw[keep], xy_index[keep]
    cells = grid['height'] * grid['width']
    index = xy_index[:, 1] * grid['width'] + xy_index[:, 0]
    count = torch.zeros(cells, device=device).index_add(0, index, torch.ones(len(index), device=device))
    cell_xy = xy_index * grid['cell_size'] + xyz.new_tensor([grid['x_min'], grid['y_min']])
    offsets = (xyz[:, :2] - cell_xy) / grid['cell_size'] - .5
    attributes = torch.nan_to_num(raw[:, 3:5]).clamp(-150, 150)
    values = torch.cat((offsets, xyz[:, 2:3] / 3, attributes[:, :1] / 30,
                        attributes[:, 1:] / 50, (xyz[:, 2:3] / 3).square(),
                        torch.linalg.vector_norm(xyz[:, :2], dim=1)[:, None] / 65), 1)
    means = torch.zeros((cells, 7), device=device).index_add(0, index, values) / count[:, None].clamp_min(1)
    variance = (means[:, 5] - means[:, 2].square()).clamp_min(0)
    tokens = torch.cat((torch.log1p(count[:, None]).clamp_max(4) / 4, means[:, :5], variance[:, None], means[:, 6:7]), 1)
    return tokens.T.reshape(8, grid['height'], grid['width']), (count > 0).float().reshape(1, grid['height'], grid['width'])


def aligned_features(camera, sparse, radar, grid, camera_from_lidar, K, lidar_from_radar, width=1920, height=1200):
    lidar, lsupport = lidar_bev(sparse, grid)
    image, csupport = camera_bev(camera, grid, camera_from_lidar, K, width, height)
    returns, rsupport = radar_bev(radar, lidar_from_radar, grid, device=camera.device)
    return dict(camera=image, lidar=lidar, radar=returns,
                availability=torch.cat((csupport, lsupport, rsupport), 0))
