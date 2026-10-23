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
    p.add_argument('--middle', action='store_true', help='Also require independent middle fusion detections')
    p.add_argument('--enhanced-scene', action='store_true', help='Check oriented boxes, distance rings and the proposal layer')
    args = p.parse_args()
    rclpy.init()
    node = Node('group8_validation_observer')
    counts, children, ids, errors, metrics = Counter(), set(), set(), [], []
    marker_namespaces = set()
    colored_points = 0

    def observe(name, msg):
        nonlocal colored_points
        counts[name] += 1
        if name in ('objects', 'detector_objects'):
            if msg.header.frame_id != 'velodyne':
                errors.append('Wrong object frame')
            for detection in msg.detections:
                if name == 'objects':
                    ids.add(detection.id)
                elif detection.id:
                    errors.append('Independent detection received an invented persistent ID')
                if name == 'detector_objects':
                    counts['independent_detections'] += 1
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
        elif name == 'instance_points':
            colored_points += msg.width
            if msg.header.frame_id != 'velodyne' or msg.point_step != 16 or 'rgb' not in [f.name for f in msg.fields]:
                errors.append('Invalid instance point cloud schema')
        elif name in ('markers', 'proposals', 'boxes', 'motion'):
            for marker in msg.markers:
                if marker.action != marker.ADD:
                    continue
                marker_namespaces.add(marker.ns)
                if not all(math.isfinite(v) for p in marker.points for v in (p.x, p.y, p.z)):
                    errors.append('Nonfinite marker geometry')
                if marker.ns in ('box_outlines', 'unfused_lidar_boxes', 'middle_fusion_boxes') and len(marker.points) != 24:
                    errors.append('Invalid cuboid edge count')

    subscriptions = []
    for typ, topic, name in [(Image, '/fusion/image', 'image'),
                             (PointCloud2, '/fusion/nonground', 'cloud'),
                             (Detection3DArray, '/fusion/objects', 'objects'),
                             (MarkerArray, '/fusion/markers', 'markers'),
                             (String, '/fusion/tracks', 'tracks'),
                             (String, '/fusion/metrics', 'metrics')]:
        qos = qos_profile_sensor_data if typ in (Image, PointCloud2) else 10
        subscriptions.append(node.create_subscription(typ, topic, lambda msg, n=name: observe(n, msg), qos))
    if args.middle:
        subscriptions.append(node.create_subscription(Detection3DArray, '/fusion/detector_objects', lambda msg: observe('detector_objects', msg), 10))
    if args.enhanced_scene:
        for topic, name in (('/fusion/proposals', 'proposals'), ('/fusion/boxes', 'boxes'), ('/fusion/motion', 'motion')):
            subscriptions.append(node.create_subscription(MarkerArray, topic, lambda msg, n=name: observe(n, msg), 10))
        subscriptions.append(node.create_subscription(PointCloud2, '/fusion/instance_points', lambda msg: observe('instance_points', msg), qos_profile_sensor_data))
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
    if args.enhanced_scene:
        required = {'box_outlines', 'box_footprints', 'track_labels', 'distance_rings',
                    'middle_fusion_boxes' if args.middle else 'unfused_lidar_boxes'}
        if required - marker_namespaces:
            errors.append(f'Missing scene layers: {sorted(required - marker_namespaces)}')
        if not colored_points:
            errors.append('No colored object points received')
    if args.middle:
        if not counts['independent_detections']:
            errors.append('No independent fused detections received')
        if not metrics or any(row.get('pipeline') != 'middle' for row in metrics):
            errors.append('Expected middle pipeline metrics throughout')
    rviz_subscriptions = {}
    if args.rviz:
        topics = ['/fusion/markers', '/fusion/image', '/velodyne/points_raw', '/PointCloudDetection']
        if args.enhanced_scene:
            topics.append('/fusion/instance_points')
        for topic in topics:
            rviz_subscriptions[topic] = any(info.node_name == 'rviz'
                for info in node.get_subscriptions_info_by_topic(topic))
            if not rviz_subscriptions[topic]:
                errors.append(f'RViz is not subscribed to {topic}')
    report = {'passed': not errors, 'counts': dict(counts), 'static_children': sorted(children),
              'unique_track_ids': len(ids), 'errors': errors, 'rviz_subscriptions': rviz_subscriptions,
              'last_metrics': metrics[-1] if metrics else None}
    report['marker_namespaces'] = sorted(marker_namespaces)
    report['colored_point_observations'] = colored_points
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
