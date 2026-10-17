#!/usr/bin/env python3
"""Observe a live demo and fail if its essential outputs are missing or invalid."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import MarkerArray


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--duration', type=float, default=65)
    p.add_argument('--output', default='/results/ros_smoke.json')
    p.add_argument('--rviz', action='store_true', help='Also require the real RViz subscriptions')
    p.add_argument('--min-images', type=int, default=1)
    args = p.parse_args()
    rclpy.init()
    node = Node('group8_validation_observer')
    counts, children, ids, errors, metrics = Counter(), set(), set(), [], []

    def observe(name, msg):
        counts[name] += 1
        if name == 'objects':
            if msg.header.frame_id != 'velodyne':
                errors.append('Wrong object frame')
            for detection in msg.detections:
                ids.add(detection.id)
                point = detection.bbox.center.position
                if not all(math.isfinite(v) for v in [point.x, point.y, point.z]):
                    errors.append('Nonfinite object coordinates')
        elif name == 'metrics':
            metrics.append(json.loads(msg.data))
        elif name == 'tracks':
            data = json.loads(msg.data)
            for track in data['tracks']:
                if not all(math.isfinite(v) for v in track['velocity_relative_xy']):
                    errors.append('Nonfinite track velocity')
        elif name == 'tf_static':
            children.update(t.child_frame_id for t in msg.transforms)
        elif name == 'image' and msg.encoding != 'bgr8':
            errors.append('Unexpected annotation encoding')

    subscriptions = []
    for typ, topic, name in [(Image, '/fusion/image', 'image'),
                             (PointCloud2, '/fusion/nonground', 'cloud'),
                             (Detection3DArray, '/fusion/objects', 'objects'),
                             (MarkerArray, '/fusion/markers', 'markers'),
                             (String, '/fusion/tracks', 'tracks'),
                             (String, '/fusion/metrics', 'metrics')]:
        qos = qos_profile_sensor_data if typ in (Image, PointCloud2) else 10
        subscriptions.append(node.create_subscription(typ, topic, lambda msg, n=name: observe(n, msg), qos))
    subscriptions.append(node.create_subscription(TFMessage, '/tf_static', lambda msg: observe('tf_static', msg),
        QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)))
    end = time.monotonic() + args.duration
    while rclpy.ok() and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    missing = sorted({'image', 'cloud', 'objects', 'tracks', 'markers', 'metrics', 'tf_static'} - counts.keys())
    if missing:
        errors.append(f'Missing output topics: {missing}')
    if counts['image'] < args.min_images:
        errors.append(f'Only {counts["image"]} images, expected at least {args.min_images}')
    if not {'cam0', 'radar', 'ouster/imu_link'}.issubset(children):
        errors.append('Missing static sensor frames')
    if not ids:
        errors.append('No confirmed tracks observed')
    rviz_subscriptions = {}
    if args.rviz:
        for topic in ('/fusion/markers', '/fusion/image', '/fusion/nonground',
                      '/velodyne/points_raw', '/PointCloudDetection'):
            rviz_subscriptions[topic] = any(info.node_name == 'rviz'
                for info in node.get_subscriptions_info_by_topic(topic))
            if not rviz_subscriptions[topic]:
                errors.append(f'RViz is not subscribed to {topic}')
    report = {'passed': not errors, 'counts': dict(counts), 'static_children': sorted(children),
              'unique_track_ids': len(ids), 'errors': errors, 'rviz_subscriptions': rviz_subscriptions,
              'last_metrics': metrics[-1] if metrics else None}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    node.destroy_node()
    rclpy.shutdown()
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
