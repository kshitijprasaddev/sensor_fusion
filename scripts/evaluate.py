#!/usr/bin/env python3
"""Replay every camera frame through the same processor as the ROS node.

Metrics measure execution and associations, not accuracy without ground truth.
"""
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import time
import cv2
import numpy as np
from group8_fusion.bag import Bag, CAMERA, LIDAR, RADAR
from group8_fusion.geometry import image_array, cloud_array
from group8_fusion.processor import load_config, CameraDetector, make_processor


def main():
    p = argparse.ArgumentParser()
    p.add_argument('bag')
    p.add_argument('--config', default='src/group8_fusion/config/fusion.yaml')
    p.add_argument('--output', default='results/evaluation')
    p.add_argument('--weights')
    p.add_argument('--device')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--modality', choices=['all', 'lidar', 'camera_lidar', 'lidar_radar'], default='all')
    p.add_argument('--video', action='store_true')
    args = p.parse_args()
    if args.stride < 1 or args.limit < 0:
        p.error('stride must be positive and limit nonnegative')
    cfg = load_config(args.config)
    if cfg.get('pipeline') in ('neural', 'middle') and args.modality != 'all':
        p.error('Neural profile supports --modality all; use the baseline config for classical ablations')
    if args.weights:
        cfg['model']['weights'] = args.weights
    if args.device:
        cfg['model']['device'] = args.device
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    bag = Bag(args.bag)
    detector = CameraDetector(cfg['model']) if args.modality in ('all', 'camera_lidar') else None
    torch_runtime = None
    if cfg.get('pipeline') in ('neural', 'middle'):
        import torch as torch_runtime
        torch_runtime.cuda.reset_peak_memory_stats()
    processor = make_processor(cfg, bag.lidar_from('cam0'), bag.lidar_from('radar'), detector)
    rows, track_rows, detector_rows = [], [], []
    video = None
    skipped = 0
    tolerance = int(cfg['synchronization']['tolerance_ms'] * 1e6)
    started = time.perf_counter()
    for i in range(0, len(bag.index[CAMERA]), args.stride):
        if args.limit and len(rows) >= args.limit:
            break
        timestamp, msg = bag.read(CAMERA, i)
        lidar = bag.nearest(LIDAR, timestamp, tolerance)
        radar = bag.nearest(RADAR, timestamp, tolerance)
        if lidar is None:
            skipped += 1
            continue
        img = image_array(msg)
        result = processor.process(img, cloud_array(lidar[1], processor.lidar_fields),
                                   cloud_array(radar[1], processor.radar_fields) if radar else None,
                                   timestamp / 1e9, args.modality)
        metrics = result['metrics']
        metrics.update({'frame': i, 'lidar_delta_ms': (lidar[0] - timestamp) / 1e6,
                        'radar_delta_ms': (radar[0] - timestamp) / 1e6 if radar else None})
        rows.append(metrics)
        for detection in result.get('detector_objects', []):
            detector_rows.append(dict(frame=i, time=timestamp / 1e9,
                                      box=detection['box'].tolist(), label=detection['label'],
                                      confidence=detection['confidence']))
        for t in result['tracks']:
            track_rows.append({'frame': i, 'time': timestamp / 1e9, 'id': t.id,
                               'x': float(t.state[0]), 'y': float(t.state[1]),
                               'vx_relative': float(t.state[2]), 'vy_relative': float(t.state[3]),
                               'label': t.object.label, 'sources': t.object.sources,
                               'measured': t.measured, 'radial_velocity_sensor': t.object.radial_velocity})
        if i % 50 == 0 or args.video:
            annotated = processor.annotate(img, result)
            if i % 50 == 0:
                cv2.imwrite(str(out / f'frame_{i:04d}.jpg'), annotated)
            if args.video:
                if video is None:
                    video = cv2.VideoWriter(str(out / 'fusion_raw.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                            10 / args.stride, (960, 600))
                    if not video.isOpened():
                        raise RuntimeError('Unable to open MP4 video writer')
                video.write(cv2.resize(annotated, (960, 600)))
        if len(rows) % 25 == 0:
            print(f'{len(rows)} frames, camera matches={sum(r["camera_matches"] for r in rows)}, '
                  f'radar matches={sum(r["radar_matches"] for r in rows)}', flush=True)
    if video:
        video.release()
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                        '-i', str(out / 'fusion_raw.mp4'), '-c:v', 'libx264', '-preset', 'fast',
                        '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                        str(out / 'fusion.mp4')], check=True)
        (out / 'fusion_raw.mp4').unlink()
    bag.close()
    if not rows:
        raise RuntimeError('No synchronized camera/LiDAR frames processed')
    with (out / 'frames.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'tracks.json').write_text(json.dumps(track_rows, indent=2) + '\n')
    if cfg.get('pipeline') == 'middle':
        (out / 'detector_objects.json').write_text(json.dumps(detector_rows, indent=2) + '\n')
    weight = Path(cfg['model']['weights'])
    summary = {'bag': Path(args.bag).name, 'modality': args.modality, 'frames': len(rows),
               'skipped_no_lidar': skipped, 'wall_time_s': time.perf_counter() - started,
               'calibration_status': cfg['camera']['status'], 'timestamp_policy': 'bag_receipt_time',
               'configuration': cfg, 'weights_sha256': hashlib.sha256(weight.read_bytes()).hexdigest() if detector else None,
               'latency_ms': dict(zip(['p50', 'p95', 'max'], map(float, np.percentile(
                   [r['latency_ms'] for r in rows], [50, 95, 100])))),
               'totals': {k: sum(r[k] for r in rows) for k in ['camera_detections', 'lidar_objects',
                   'radar_objects', 'camera_matches', 'radar_matches', 'three_sensor_objects']},
               'unique_confirmed_tracks': len({r['id'] for r in track_rows}),
               'pipeline': cfg.get('pipeline', 'classical'),
               'fusion_schema': getattr(processor, 'fusion_schema', None),
               'fusion_weights_sha256': getattr(processor, 'checkpoint_sha256', None),
               'middle_detector_weights_sha256': getattr(processor, 'middle_checkpoint_sha256', None),
               'independent_detector_objects': len(detector_rows) if cfg.get('pipeline') == 'middle' else None,
               'radar_count_unit': 'filtered_points' if cfg.get('pipeline') in ('neural', 'middle') else 'clusters',
               'torch_peak_allocated_mib': torch_runtime.cuda.max_memory_allocated() / 1024 ** 2 if torch_runtime else None,
               'memory_boundary': 'PyTorch allocator including encoders, SAM, association and middle head; excludes TensorRT and driver allocations' if torch_runtime else None,
               'accuracy_metrics': 'Unavailable: no labeled ground truth, true camera intrinsics or ego pose supplied.'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'configuration'}, indent=2))


if __name__ == '__main__':
    main()
