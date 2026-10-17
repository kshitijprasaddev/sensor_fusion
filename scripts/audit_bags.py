#!/usr/bin/env python3
"""Extract verifiable topic, timing and extrinsic evidence from real bags."""
import argparse
import json
from pathlib import Path
import numpy as np
from group8_fusion.bag import Bag, CAMERA, LIDAR, RADAR


def audit(path):
    bag = Bag(path)
    result = {'name': bag.path.parent.name, 'duration_s': (bag.end - bag.start) / 1e9,
              'message_count': sum(len(v) for v in bag.index.values()), 'topics': {},
              'camera_intrinsics_present': any('CameraInfo' in t[2] for t in bag.topics.values()),
              'transforms': []}
    for name, entries in bag.index.items():
        count = len(entries)
        item = {'type': bag.topics[name][2], 'count': count}
        if count > 1:
            item['receipt_rate_hz'] = (count - 1) / ((entries[-1, 1] - entries[0, 1]) / 1e9)
        offsets = []
        for i in np.linspace(0, count - 1, min(12, count), dtype=int):
            ts, msg = bag.read(name, int(i))
            if hasattr(msg, 'header'):
                offsets.append((ts - (msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec)) / 1e9)
                item['frame_id'] = msg.header.frame_id
                if hasattr(msg, 'fields'):
                    item['fields'] = [f.name for f in msg.fields]
        if offsets:
            item['sample_receipt_minus_header_s'] = {'min': min(offsets), 'max': max(offsets),
                                                     'median': float(np.median(offsets))}
        result['topics'][name] = item
    for t in bag.static_transforms():
        p, q = t.transform.translation, t.transform.rotation
        result['transforms'].append({'parent': t.header.frame_id, 'child': t.child_frame_id,
                                      'translation': [p.x, p.y, p.z], 'quaternion_xyzw': [q.x, q.y, q.z, q.w]})
    result['nearest_receipt_delta_ms'] = {}
    for name in (LIDAR, RADAR):
        stamps = bag.index[name][:, 1]
        deltas = [float(np.min(np.abs(stamps - ts))) / 1e6 for ts in bag.index[CAMERA][:, 1]]
        result['nearest_receipt_delta_ms'][name] = dict(zip(['p50', 'p95', 'max'],
                                                           map(float, np.percentile(deltas, [50, 95, 100]))))
    bag.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bags', nargs='+')
    parser.add_argument('--output', default='results/dataset_audit.json')
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([audit(p) for p in args.bags], indent=2) + '\n')
    print(out)
