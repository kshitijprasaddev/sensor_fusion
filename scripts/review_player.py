#!/usr/bin/env python3
"""Replay exported sensor/model stages for explanation, never as a speed benchmark."""
import argparse
from array import array
import json
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header, String
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from group8_fusion.geometry import oriented_corners, BOX_EDGES
from group8_fusion.bag import Bag
from tf2_ros import StaticTransformBroadcaster


def point_cloud(header, data, names):
    data = np.asarray(data, np.float32).reshape(-1, len(names))
    data = data[np.isfinite(data[:, :3]).all(1)]
    return PointCloud2(header=header, height=1, width=len(data), is_bigendian=False,
                      fields=[PointField(name=n, offset=i*4, datatype=PointField.FLOAT32, count=1) for i, n in enumerate(names)],
                      point_step=4*len(names), row_step=4*len(names)*len(data),
                      data=array('B', data.astype('<f4').tobytes()), is_dense=True)


def panel(mode, row, data, total):
    img = np.full((800, 1120, 3), (28, 23, 18), np.uint8)
    ink, muted, accent = (240, 240, 240), (185, 180, 170), (220, 205, 90)
    def text(s, y, size=.8, color=ink):
        cv2.putText(img, s, (36, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 2, cv2.LINE_AA)
    text('LiDAR processing' if mode == 'lidar' else 'Radar processing', 65, 1.15)
    text(f"Recorded frame {row['frame'] + 1} / {total}", 110, .75, muted)
    text('Replay of actual model outputs', 150, .7, muted)
    if mode == 'lidar':
        lines = [f"1  Read XYZ + intensity: {row['lidar_raw_points']:,} points",
                 f"2  Normalize intensity and keep model ROI",
                 f"   {row['lidar_prepared_points']:,} prepared points",
                 f"3  Build {row['lidar_voxels']:,} occupied voxels",
                 '4  Pretrained sparse VoxelNeXt network',
                 f"5  {row['lidar_proposals']} 3D proposals at score >= 0.25",
                 f"   128 learned features per proposal"]
        for i, line in enumerate(lines):
            text(line, 225 + i*58, .82, accent if i in (0, 5) else ink)
        text('Grey: raw returns   Cyan: prepared model points', 675, .68, muted)
        text('Boxes: LiDAR proposals, before sensor association', 720, .68, muted)
    else:
        lines = [f"1  Read {row['radar_raw_returns']} recorded detection points",
                 '   XYZ + recorded velocity + RCS',
                 '2  Transform radar coordinates into velodyne',
                 f"3  Retain {row['radar_retained_returns']} finite / in-range returns",
                 f"4  Encode {row['radar_token_shape'][1]} values per return",
                 '5  Learned MLP and contextual transformer',
                 '   Features then enter query cross-attention']
        for i, line in enumerate(lines):
            text(line, 225 + i*58, .80, accent if i in (0, 3) else ink)
        text('Colour: recorded velocity attribute, range -10 to +10', 675, .64, muted)
        text('Radar points are not individual detected vehicles', 720, .68, muted)
    text('Coordinates follow the moving rig. No global map.', 770, .64, muted)
    return img


class ReviewPlayer(Node):
    def __init__(self, args):
        super().__init__('group8_review_player')
        self.args = args
        self.manifest = json.loads((args.cache/'manifest.json').read_text())
        if self.manifest.get('schema') != 'group8-review-v1':
            raise ValueError('Unsupported cache')
        self.rows = self.manifest['rows']
        if len(self.rows) != self.manifest['frames']:
            raise ValueError('Cache frame count differs from manifest')
        self.index = 0
        self.raw = self.create_publisher(PointCloud2, '/review/raw_points', qos_profile_sensor_data)
        self.processed = self.create_publisher(PointCloud2, '/review/processed_points', qos_profile_sensor_data)
        self.boxes = self.create_publisher(MarkerArray, '/review/boxes', 2)
        self.image = self.create_publisher(Image, '/review/image', qos_profile_sensor_data)
        self.metrics = self.create_publisher(String, '/review/metrics', 2)
        self.clock = self.create_publisher(Clock, '/clock', 2)
        bag = Bag(args.bag)
        self.static_tf = StaticTransformBroadcaster(self)
        self.transforms = bag.static_transforms()
        bag.close()
        for transform in self.transforms:
            transform.header.stamp.sec = 0
            transform.header.stamp.nanosec = 0
        self.static_tf.sendTransform(self.transforms)
        self.create_timer(1/args.fps, self.tick)
        self.get_logger().info(f'Cached {args.mode} diagnostic replay at {args.fps:g} FPS. This is not live inference.')

    def tick(self):
        row = self.rows[self.index]
        header = Header(frame_id='velodyne')
        header.stamp.sec, header.stamp.nanosec = divmod(row['time_ns'], 10**9)
        self.clock.publish(Clock(clock=header.stamp))
        # RViz clears its TF buffer when the recorded clock loops backwards.
        # Repeat on the next tick too, after subscribers handle that reset.
        if self.index < 2:
            self.static_tf.sendTransform(self.transforms)
        with np.load(self.args.cache/'frames'/f'{self.index:05}.npz', allow_pickle=False) as data:
            markers = [Marker(action=Marker.DELETEALL)]
            if self.args.mode == 'camera':
                img = cv2.imread(str(self.args.cache/'frames'/f'{self.index:05}_camera.jpg'))
                if img is None:
                    raise RuntimeError('Camera frame missing')
            elif self.args.mode == 'lidar':
                self.raw.publish(point_cloud(header, data['lidar_raw'][::2], ['x','y','z','intensity']))
                self.processed.publish(point_cloud(header, data['lidar_input'][:, :4], ['x','y','z','intensity']))
                for i, (box, score, label) in enumerate(zip(data['lidar_boxes'], data['lidar_scores'], data['lidar_labels'])):
                    corners = oriented_corners(box[:3], box[3:6], box[6])
                    marker = Marker(header=header, ns='lidar_proposals', id=i*2, type=Marker.LINE_LIST, action=Marker.ADD)
                    marker.pose.orientation.w = 1.; marker.scale.x = .035
                    marker.color.r, marker.color.g, marker.color.b, marker.color.a = .35, .88, .78, .95
                    marker.points = [Point(x=float(corners[j,0]), y=float(corners[j,1]), z=float(corners[j,2])) for edge in BOX_EDGES for j in edge]
                    markers.append(marker)
                    label_marker = Marker(header=header, ns='lidar_proposals', id=i*2+1, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
                    label_marker.pose.position = Point(x=float(box[0]), y=float(box[1]), z=float(box[2]+box[5]/2+.35))
                    label_marker.pose.orientation.w = 1.; label_marker.scale.z = .28
                    label_marker.color.r = label_marker.color.g = label_marker.color.b = label_marker.color.a = 1.
                    label_marker.text = f"{self.manifest['lidar_classes'][int(label)]} {score:.2f}"
                    markers.append(label_marker)
                img = panel('lidar', row, data, len(self.rows))
            else:
                self.raw.publish(point_cloud(header, data['radar_raw'], ['x','y','z','v','RCS']))
                shown = np.c_[data['radar_xyz'], data['radar_v']]
                self.processed.publish(point_cloud(header, shown, ['x','y','z','v']))
                img = panel('radar', row, data, len(self.rows))
            self.boxes.publish(MarkerArray(markers=markers))
            self.image.publish(Image(header=header, height=img.shape[0], width=img.shape[1], encoding='bgr8',
                                     step=img.shape[1]*3, data=array('B', img.tobytes())))
        self.metrics.publish(String(data=json.dumps(dict(mode=self.args.mode, cached_replay=True, **row))))
        self.index = (self.index+1) % len(self.rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['camera','lidar','radar'])
    parser.add_argument('--cache', type=Path, default=Path('/results/review_2026_11_26'))
    parser.add_argument('--bag', default='/data/studentProject')
    parser.add_argument('--fps', type=float, default=10)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error('FPS must be positive')
    rclpy.init(); node = ReviewPlayer(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.try_shutdown()


if __name__ == '__main__':
    main()
