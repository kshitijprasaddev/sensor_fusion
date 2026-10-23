#!/usr/bin/env python3
"""GPU integration guards for independent middle fusion and optional backward wiring."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch
from group8_fusion import perception, tracking, processor
from group8_fusion.bag import Bag, CAMERA, LIDAR, RADAR
from group8_fusion.geometry import cloud_array, image_array
from group8_fusion.processor import load_config, CameraDetector, make_processor
from group8_fusion.mid_features import CameraSpatialEncoder, LidarSpatialEncoder, aligned_features, GRID
from group8_fusion.middle import load_middle_checkpoint
from group8_fusion.mid_training import center_targets, teacher_targets, detection_loss


def forbidden(*args, **kwargs):
    raise AssertionError('A forbidden teacher detection or classical inference path was executed')


def gradient_report(module):
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    return dict(tensors_with_gradients=len(gradients),
                all_finite=all(bool(torch.isfinite(value).all()) for value in gradients),
                absolute_gradient_sum=sum(float(value.abs().sum()) for value in gradients))


def backward_check(cfg, bag, frame):
    """One backward pass proves wiring, without optimizer steps or training claims."""
    if not Path(f'/results/middle_features/studentProject/{frame:05d}.npz').is_file():
        raise FileNotFoundError('The optional backward check needs local teacher targets: run mid-extract first')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    camera = CameraSpatialEncoder(cfg['middle']['camera_weights'])
    lidar = LidarSpatialEncoder(cfg['neural']['lidar'])
    camera.requires_grad_(True).eval()
    lidar.requires_grad_(True)
    # spconv only saves convolution backward buffers in training mode. Keep
    # normalization statistics fixed while enabling sparse convolution gradients.
    lidar.network.train()
    for module in lidar.network.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    lidar.network.dense_head.forward = forbidden
    camera.network.model[23].forward = forbidden
    model, _, _ = load_middle_checkpoint(cfg['middle']['weights'], cfg)
    model = model.cuda().train()
    timestamp, message = bag.read(CAMERA, frame)
    raw_lidar = cloud_array(bag.nearest(LIDAR, timestamp)[1], ('x', 'y', 'z', 'intensity'))
    raw_radar = cloud_array(bag.nearest(RADAR, timestamp)[1], ('x', 'y', 'z', 'v', 'RCS'))
    image_tensor = camera.tensor(image_array(message)).requires_grad_(True)
    radar_tensor = torch.as_tensor(raw_radar.copy(), device='cuda').requires_grad_(True)
    features = aligned_features(camera(image_tensor), lidar(raw_lidar), radar_tensor, GRID,
                                np.linalg.inv(bag.lidar_from('cam0')), cfg['camera']['K'], bag.lidar_from('radar'))
    prediction = model(**{key: value[None] for key, value in features.items()})
    with np.load(f'/results/middle_features/studentProject/{frame:05d}.npz') as cache:
        boxes, labels, scores = teacher_targets(cache['teacher_boxes'], cache['teacher_labels'], cache['teacher_scores'], GRID)
    target = center_targets(boxes, labels, scores, GRID)
    loss, _ = detection_loss(prediction, {key: target[key][None].cuda() for key in ('heatmap', 'regression', 'mask', 'weight')})
    loss.backward()
    early_gradient = lidar.network.backbone_3d.conv_input[0].weight.grad
    assert early_gradient is not None and torch.isfinite(early_gradient).all() and early_gradient.abs().sum() > 0
    report = dict(loss=float(loss.detach()),
                  lidar_first_sparse_convolution_gradient_sum=float(early_gradient.abs().sum()), camera_encoder=gradient_report(camera),
                  lidar_encoder=gradient_report(lidar), middle_fusion_head=gradient_report(model),
                  image_pixel_gradient_sum=float(image_tensor.grad.abs().sum()),
                  radar_attribute_gradient_sum=float(radar_tensor.grad[:, 3:].abs().sum()),
                  radar_gradients_finite=bool(torch.isfinite(radar_tensor.grad).all()),
                  peak_torch_allocated_mib=torch.cuda.max_memory_allocated() / 1024 ** 2,
                  optimizer_steps=0,
                  meaning='One-frame backward integration proof only. Published encoders remain frozen in the delivered 90-epoch training. LiDAR voxelization is discrete; raw point coordinate gradients are not claimed.')
    for key in ('camera_encoder', 'lidar_encoder', 'middle_fusion_head'):
        assert report[key]['all_finite'] and report[key]['absolute_gradient_sum'] > 0, key
    assert report['image_pixel_gradient_sum'] > 0
    assert report['radar_gradients_finite'] and report['radar_attribute_gradient_sum'] > 0
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames', type=int, default=5)
    p.add_argument('--backward', action='store_true')
    p.add_argument('--output', default='/results/middle_runtime_guard.json')
    a = p.parse_args()
    if a.frames < 1:
        p.error('--frames must be positive')
    torch.set_num_threads(2)
    for name in ('dbscan', 'remove_ground', 'lidar_detect', 'radar_detect', 'associate_camera', 'associate_radar', 'linear_sum_assignment'):
        setattr(perception, name, forbidden)
    for name in ('lidar_detect', 'radar_detect', 'associate_camera', 'associate_radar'):
        setattr(processor, name, forbidden)
    tracking.Tracker.update = forbidden
    tracking.linear_sum_assignment = forbidden
    cfg = load_config('src/group8_fusion/config/fusion_middle.yaml')
    cfg['model']['weights'] = '/models/yolo11n.engine'
    bag = Bag('/data/studentProject')
    node = make_processor(cfg, bag.lidar_from('cam0'), bag.lidar_from('radar'), CameraDetector(cfg['model']))
    node.spatial_lidar.network.dense_head.forward = forbidden
    node.lidar_detector.detect = forbidden
    node.spatial_camera.network.model[23].forward = forbidden
    rows = []
    cache_parity = {}
    spatial = None
    for i in range(min(a.frames, len(bag.index[CAMERA]))):
        timestamp, msg = bag.read(CAMERA, i)
        l, r = bag.nearest(LIDAR, timestamp), bag.nearest(RADAR, timestamp)
        if i == 0 and Path('/results/middle_features/studentProject/00000.npz').is_file():
            with torch.inference_mode():
                spatial = aligned_features(node.spatial_camera(node.spatial_camera.tensor(image_array(msg))),
                    node.spatial_lidar(cloud_array(l[1], node.lidar_fields)),
                    cloud_array(r[1], node.radar_fields), GRID, node.camera_from_lidar, node.K, node.lidar_from_radar)
            with np.load('/results/middle_features/studentProject/00000.npz') as cached:
                for key, value in spatial.items():
                    actual = value.cpu().numpy()
                    expected = cached[key].astype(np.float32)
                    cache_parity[key] = dict(max_absolute_difference=float(np.max(np.abs(actual - expected))),
                                             passed=bool(np.allclose(actual, expected, rtol=.002, atol=.01)))
                    assert cache_parity[key]['passed'], f'Cached and runtime features differ: {key}'
        result = node.process(image_array(msg), cloud_array(l[1], node.lidar_fields),
                              cloud_array(r[1], node.radar_fields), timestamp / 1e9)
        assert result['metrics']['pipeline'] == 'middle'
        assert all(np.isfinite(d['box']).all() and (d['box'][3:6] > 0).all() for d in result['detector_objects'])
        rows.append(dict(frame=i, independent_detections=len(result['detector_objects']),
                         tracks=len(result['tracks']), metrics=result['metrics']))
    assert sum(row['independent_detections'] for row in rows) > 0
    # Loss of radar remains a legal frame; no stale radar features are reused.
    node.reset_tracking()
    missing = node.process(image_array(msg), cloud_array(l[1], node.lidar_fields), None, timestamp / 1e9 + 1)
    assert missing['metrics']['radar_objects'] == 0 and missing['metrics']['middle_radar_gate'] == 0
    node.reset_tracking()
    empty = node.process(image_array(msg), np.empty((0, 4), np.float32), None, timestamp / 1e9 + 2)
    assert empty['metrics']['middle_lidar_gate'] == 0
    assert empty['metrics']['middle_radar_gate'] == 0
    report = dict(passed=True, checkpoint_sha256=node.middle_checkpoint_sha256,
                  forbidden_classical_calls=0, teacher_detection_head_calls=0,
                  camera_detection_head_calls_in_spatial_encoder=0,
                  missing_radar_passed=True, empty_lidar_passed=True,
                  cached_runtime_feature_parity=cache_parity, observations=rows)
    del node, missing, empty, spatial
    gc.collect()
    torch.cuda.empty_cache()
    if a.backward:
        report['backward_integration'] = backward_check(cfg, bag, 0)
    bag.close()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
