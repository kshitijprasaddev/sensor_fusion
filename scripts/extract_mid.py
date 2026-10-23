#!/usr/bin/env python3
"""Cache spatial encoder features and separate teacher-only supervision."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from group8_fusion.bag import Bag, CAMERA, LIDAR, RADAR
from group8_fusion.geometry import image_array, cloud_array
from group8_fusion.processor import load_config
from group8_fusion.mid_features import CameraSpatialEncoder, LidarSpatialEncoder, aligned_features, GRID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='/results/middle_features')
    parser.add_argument('--config', default='src/group8_fusion/config/fusion_ai.yaml')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--bags', nargs='+', default=['studentProject', 'studentProject1'])
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be nonnegative')
    torch.set_num_threads(2)
    cfg = load_config(args.config)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    for name in args.bags:
        if list((root / name).glob('*.npz')):
            raise FileExistsError(f'Use a new cache directory: {root / name}')
    camera_weights = cfg.get('middle', {}).get('camera_weights', '/models/yolo11n.pt')
    camera = CameraSpatialEncoder(camera_weights)
    lidar_encoder = LidarSpatialEncoder(cfg['neural']['lidar'])
    manifest = dict(schema='group8-middle-features-v1', supervision='teacher_distillation',
                    configuration=cfg, grid=GRID, bags={}, model_hashes={
                        name: hashlib.sha256(Path(filename).read_bytes()).hexdigest()
                        for name, filename in [('yolo11n.pt', camera_weights),
                                               ('voxelnext_nuscenes.pth', cfg['neural']['lidar']['weights'])]})
    for name in args.bags:
        bag = Bag('/data/' + name)
        out = root / name
        out.mkdir(exist_ok=True)
        start = time.perf_counter()
        rows = []
        with torch.inference_mode():
            for index in range(len(bag.index[CAMERA])):
                if args.limit and len(rows) >= args.limit:
                    break
                timestamp, message = bag.read(CAMERA, index)
                tolerance = int(cfg['synchronization']['tolerance_ms'] * 1e6)
                lidar = bag.nearest(LIDAR, timestamp, tolerance)
                radar = bag.nearest(RADAR, timestamp, tolerance)
                if lidar is None:
                    raise RuntimeError(f'Missing synchronized LiDAR at {name}/{index}')
                before = time.perf_counter()
                image = image_array(message)
                if image.shape[1::-1] != (cfg['camera']['width'], cfg['camera']['height']):
                    raise ValueError('Image dimensions differ from camera calibration')
                raw_lidar = cloud_array(lidar[1], ('x', 'y', 'z', 'intensity'))
                raw_radar = cloud_array(radar[1], ('x', 'y', 'z', 'v', 'RCS')) if radar else np.empty((0, 5), np.float32)
                sparse = lidar_encoder(raw_lidar, teacher=True)
                features = aligned_features(camera(camera.tensor(image)), sparse, raw_radar,
                                             GRID, np.linalg.inv(bag.lidar_from('cam0')),
                                             cfg['camera']['K'], bag.lidar_from('radar'), image.shape[1], image.shape[0])
                record = {key: value.float().cpu().numpy().astype(np.float16) for key, value in features.items()}
                record.update({key: sparse[key] for key in ('teacher_boxes', 'teacher_labels', 'teacher_scores')})
                record.update(time=np.array(timestamp / 1e9), frame=np.array(index))
                if not all(np.isfinite(value).all() for value in record.values()):
                    raise FloatingPointError(f'Non-finite cache at {name}/{index}')
                np.savez_compressed(out / f'{index:05d}.npz', **record)
                rows.append(dict(frame=index, teacher_objects=len(record['teacher_boxes']),
                                 latency_ms=(time.perf_counter() - before) * 1000))
                if len(rows) % 25 == 0 or args.limit:
                    print(name, len(rows), rows[-1], flush=True)
        manifest['bags'][name] = dict(frames=len(rows), wall_time_s=time.perf_counter() - start,
                                      per_frame=rows)
        bag.close()
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v['frames'] for k, v in manifest['bags'].items()}), flush=True)


if __name__ == '__main__':
    main()
