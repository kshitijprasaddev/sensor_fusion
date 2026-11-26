#!/usr/bin/env python3
"""Export actual model stages for a reproducible, explicitly cached RViz review."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from group8_fusion.bag import Bag, CAMERA, LIDAR, RADAR
from group8_fusion.geometry import cloud_array, image_array, transform_points
from group8_fusion.processor import CameraDetector, load_config
from group8_fusion.neural import NeuralProcessor
from group8_fusion.display import track_bgr, text_tag


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', default='/data/studentProject')
    parser.add_argument('--output', type=Path, default=Path('/results/review_2026_11_26'))
    parser.add_argument('--config', default='/review-source/src/group8_fusion/config/fusion_review.yaml')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        parser.error('This export already exists. Choose a fresh --output directory.')
    frames = out / 'frames'
    frames.mkdir(exist_ok=True)
    if list(frames.glob('*.npz')):
        parser.error('Partial frame export exists. Choose a fresh --output directory.')
    cfg = load_config(args.config)
    cfg['model']['weights'] = '/models/yolo11n.engine'
    cfg['neural']['lidar']['deterministic'] = True
    assert cfg['model']['confidence'] > 0.6
    bag = Bag(args.bag)
    detector = CameraDetector(cfg['model'])
    processor = NeuralProcessor(cfg, bag.lidar_from('cam0'), bag.lidar_from('radar'), detector)
    # Observe the actual VoxelNeXt data processor, rather than inventing another ROI.
    capture = {}
    original_prepare = processor.lidar_detector.dataset.prepare_data

    def capture_prepare(*a, **kw):
        data = original_prepare(*a, **kw)
        capture['points'] = data['points'].copy()
        capture['voxel_count'] = len(data['voxel_coords'])
        return data

    processor.lidar_detector.dataset.prepare_data = capture_prepare
    radar_context = {}

    def context_hook(module, inputs, output):
        radar_context['shape'] = list(output.shape)
        radar_context['norm'] = output.detach().norm(dim=-1).cpu().numpy().reshape(-1)

    hook = processor.network.radar_context.register_forward_hook(context_hook)
    writers = {}
    for name in ('yolo_conf065', 'sam_prompts_conf065'):
        writer = cv2.VideoWriter(str(out / (name + '_raw.mp4')), cv2.VideoWriter_fourcc(*'mp4v'), 10, (1280, 800))
        if not writer.isOpened():
            raise RuntimeError('Unable to open video encoder')
        writers[name] = writer
    rows, counts = [], Counter()
    previous = set()
    tolerance = int(cfg['synchronization']['tolerance_ms'] * 1e6)
    total = len(bag.index[CAMERA]) if args.limit == 0 else min(args.limit, len(bag.index[CAMERA]))
    for i in range(total):
        timestamp, imsg = bag.read(CAMERA, i)
        lsample = bag.nearest(LIDAR, timestamp, tolerance)
        rsample = bag.nearest(RADAR, timestamp, tolerance)
        if lsample is None:
            raise RuntimeError(f'Missing LiDAR at frame {i}')
        image = image_array(imsg)
        lidar = cloud_array(lsample[1], ('x', 'y', 'z', 'intensity'))
        radar = cloud_array(rsample[1], ('x', 'y', 'z', 'v', 'RCS')) if rsample else np.empty((0, 5))
        started = time.perf_counter()
        record, visual, boxes = processor.extract(image, lidar, radar, timestamp / 1e9)
        radar_context.clear()
        tracks, extra = processor.infer_record(record)
        latency = (time.perf_counter() - started) * 1000
        raw_radar_xyz = transform_points(radar[:, :3], processor.lidar_from_radar)
        raw_radar = np.c_[raw_radar_xyz, radar[:, 3:]].astype(np.float32)
        arrays = dict(lidar_raw=lidar.astype(np.float32),
                      lidar_input=capture['points'].astype(np.float32),
                      lidar_boxes=record['boxes'], lidar_scores=record['lidar_scores'],
                      lidar_labels=record['lidar_labels'], lidar_features=record['lidar'][:, :128],
                      radar_raw=raw_radar, radar_xyz=record['radar_xyz'],
                      radar_v=record['radar_v'], radar_tokens=record['radar'])
        assert all(np.isfinite(a).all() for k, a in arrays.items() if k != 'lidar_raw' and k != 'radar_raw')
        np.savez_compressed(frames / f'{i:05}.npz', **arrays)
        yolo_image = image.copy()
        mask_image = image.copy()
        eligible = [b for b in boxes if b['label'] in ('car', 'truck', 'bus', 'person', 'motorcycle', 'bicycle')]
        for b in boxes:
            counts[b['label']] += 1
            x1, y1, x2, y2 = np.rint(b['xyxy']).astype(int)
            color = (255, 205, 70) if b in eligible else (80, 160, 255)
            cv2.rectangle(yolo_image, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
            text_tag(yolo_image, f"{b['label']} {b['confidence']:.2f}", (x1, max(35, y1 - 12)), color, .9)
        for t in visual:
            mask, color = t['mask'], track_bgr(t['id'])
            mask_image[mask] = (mask_image[mask].astype(np.float32) * .70 + np.array(color) * .30).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask_image, contours, -1, color, 3, cv2.LINE_AA)
            x1, y1 = t['bbox'][:2].astype(int)
            text_tag(mask_image, f"{t['label']} #{t['id']}", (x1, max(35, y1 - 12)), color, .9)
        yolo_image = cv2.resize(yolo_image, (1280, 800))
        mask_image = cv2.resize(mask_image, (1280, 800))
        cv2.imwrite(str(frames / f'{i:05}_camera.jpg'), mask_image)
        writers['yolo_conf065'].write(yolo_image)
        writers['sam_prompts_conf065'].write(mask_image)
        ids = {t['id'] for t in visual}
        rows.append(dict(frame=i, time_ns=timestamp, camera_boxes=len(boxes),
                         boxes=[dict(label=b['label'], confidence=b['confidence'], xyxy=b['xyxy']) for b in boxes],
                         eligible_boxes=len(eligible), visual_ids=sorted(ids), new_visible_ids=sorted(ids - previous),
                         lidar_raw_points=len(lidar), lidar_prepared_points=len(capture['points']),
                         lidar_voxels=int(capture['voxel_count']), lidar_proposals=len(record['boxes']),
                         radar_raw_returns=len(radar), radar_retained_returns=len(record['radar_xyz']),
                         radar_token_shape=list(record['radar'].shape), radar_context_shape=radar_context.get('shape'),
                         fused_observations=len(tracks), radar_matches=extra.get('radar_matches', 0),
                         core_latency_ms=latency, lidar_delta_ms=(lsample[0]-timestamp)/1e6,
                         radar_delta_ms=(rsample[0]-timestamp)/1e6 if rsample else None))
        previous = ids
        if i in (20, 100, 140, 190):
            cv2.imwrite(str(out / f'yolo_{i:03}.jpg'), yolo_image)
            cv2.imwrite(str(out / f'sam_{i:03}.jpg'), mask_image)
        if i % 25 == 0:
            print(f'Frame {i}/{total}: YOLO {len(boxes)}, SAM {len(visual)}, LiDAR {len(record["boxes"])}, radar {len(record["radar_xyz"])}', flush=True)
    hook.remove()
    for writer in writers.values():
        writer.release()
    bag.close()
    for name in writers:
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-i', str(out / (name + '_raw.mp4')),
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                        str(out / (name + '.mp4'))], check=True)
        (out / (name + '_raw.mp4')).unlink()
    manifest = dict(schema='group8-review-v1', bag=Path(args.bag).name, frames=len(rows), fps=10,
                    camera_confidence=cfg['model']['confidence'], configuration=cfg,
                    config_sha256=digest(args.config), fusion_checkpoint_sha256=processor.checkpoint_sha256,
                    lidar_classes=processor.lidar_detector.names, yolo_box_counts=dict(counts),
                    presentation_scope='Actual model outputs cached for diagnostic playback. Replay frame rate is not inference speed. No independent accuracy labels.',
                    rows=rows)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('rows', 'configuration')}, indent=2))


if __name__ == '__main__':
    main()
