"""ROS adapter with bounded synchronization queues and RViz outputs."""
from collections import deque
from array import array
import json
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header, String
from visualization_msgs.msg import MarkerArray
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from .bag import Bag, CAMERA, LIDAR, RADAR
from .geometry import cloud_array, image_array
from .processor import load_config, CameraDetector, make_processor
from .scene import SceneRenderer
from .quality import instance_points
from .display import PALETTE


def stamp_ns(msg):
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


def xyz_cloud(header, points):
    points = np.ascontiguousarray(points, dtype='<f4')
    return PointCloud2(header=header, height=1, width=len(points),
        fields=[PointField(name=name, offset=offset, datatype=PointField.FLOAT32, count=1)
                for name, offset in [('x', 0), ('y', 4), ('z', 8)]],
        is_bigendian=False, point_step=12, row_step=len(points) * 12,
        data=array('B', points.tobytes()), is_dense=bool(np.isfinite(points).all()))


def instance_cloud(header, points, identifiers):
    packed = np.empty(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<u4')])
    if len(points):
        packed['x'], packed['y'], packed['z'] = np.asarray(points).T
        colors = np.asarray(PALETTE, dtype=np.uint32)[(np.asarray(identifiers) - 1) % len(PALETTE)]
        packed['rgb'] = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
    return PointCloud2(header=header, height=1, width=len(points),
        fields=[PointField(name=name, offset=offset, datatype=PointField.FLOAT32, count=1)
                for name, offset in [('x', 0), ('y', 4), ('z', 8), ('rgb', 12)]],
        is_bigendian=False, point_step=16, row_step=len(points) * 16,
        data=array('B', packed.tobytes()), is_dense=True)


class FusionNode(Node):
    def __init__(self):
        super().__init__('group8_fusion')
        for key, default in [('config', ''), ('bag', ''), ('weights', ''), ('device', '')]:
            self.declare_parameter(key, default)
        cfg = load_config(self.get_parameter('config').value)
        for key in ('weights', 'device'):
            value = self.get_parameter(key).value
            if value:
                cfg['model'][key] = value
        bag = Bag(self.get_parameter('bag').value)
        self.processor = make_processor(cfg, bag.lidar_from('cam0'), bag.lidar_from('radar'), CameraDetector(cfg['model']))
        bag.close()
        self.tolerance = int(cfg['synchronization']['tolerance_ms'] * 1e6)
        self.clouds = {LIDAR: deque(maxlen=40), RADAR: deque(maxlen=60)}
        self.images = deque(maxlen=4)
        self.last_stamp = None
        self.stats = {'processed': 0, 'dropped_no_lidar': 0, 'dropped_images': 0}
        self.image_pub = self.create_publisher(Image, '/fusion/image', qos_profile_sensor_data)
        self.cloud_pub = self.create_publisher(PointCloud2, '/fusion/nonground', qos_profile_sensor_data)
        self.instance_pub = self.create_publisher(PointCloud2, '/fusion/instance_points', qos_profile_sensor_data)
        self.marker_pub = self.create_publisher(MarkerArray, '/fusion/markers', 2)
        self.box_pub = self.create_publisher(MarkerArray, '/fusion/boxes', 2)
        self.motion_pub = self.create_publisher(MarkerArray, '/fusion/motion', 2)
        self.proposal_pub = self.create_publisher(MarkerArray, '/fusion/proposals', 2)
        self.scene = SceneRenderer(cfg.get('visualization', {}))
        self.detection_pub = self.create_publisher(Detection3DArray, '/fusion/objects', 2)
        self.track_pub = self.create_publisher(String, '/fusion/tracks', 2)
        self.metrics_pub = self.create_publisher(String, '/fusion/metrics', 10)
        self.create_subscription(Image, CAMERA, self.on_image, qos_profile_sensor_data)
        for topic in (LIDAR, RADAR):
            self.create_subscription(PointCloud2, topic, lambda msg, t=topic: self.on_cloud(t, msg), qos_profile_sensor_data)
        # Wall timer stays responsive across /clock rewinds when looping the bag.
        from rclpy.clock import Clock, ClockType
        self.create_timer(0.04, self.process_pending, clock=Clock(clock_type=ClockType.STEADY_TIME))
        if cfg['camera']['status'] != 'calibrated':
            self.get_logger().warning('APPROXIMATE camera intrinsics. Fusion geometry is unverified; any enabled image projection is diagnostic only.')

    def on_cloud(self, topic, msg):
        queue = self.clouds[topic]
        if queue and stamp_ns(msg) < stamp_ns(queue[-1]):
            queue.clear()
        queue.append(msg)

    def on_image(self, msg):
        timestamp = stamp_ns(msg)
        if self.last_stamp is not None and timestamp <= self.last_stamp:
            self.images.clear()
            for queue in self.clouds.values():
                queue.clear()
            self.processor.reset_tracking()
        self.last_stamp = timestamp
        if len(self.images) == self.images.maxlen:
            self.stats['dropped_images'] += 1
        self.images.append((time.monotonic(), msg))

    def nearest(self, topic, timestamp):
        queue = self.clouds[topic]
        if not queue:
            return None
        msg = min(queue, key=lambda m: abs(stamp_ns(m) - timestamp))
        return msg if abs(stamp_ns(msg) - timestamp) <= self.tolerance else None

    def process_pending(self):
        if not self.images:
            return
        received, msg = self.images[0]
        timestamp = stamp_ns(msg)
        latest = self.clouds[LIDAR][-1] if self.clouds[LIDAR] else None
        # Allow future cloud samples to arrive. Bound the wait at low replay rates.
        if time.monotonic() - received < 0.5 and (latest is None or stamp_ns(latest) < timestamp):
            return
        self.images.popleft()
        lidar, radar = self.nearest(LIDAR, timestamp), self.nearest(RADAR, timestamp)
        if lidar is None:
            self.stats['dropped_no_lidar'] += 1
            return
        node_started = time.perf_counter()
        image = image_array(msg)
        result = self.processor.process(image, cloud_array(lidar, self.processor.lidar_fields),
            cloud_array(radar, self.processor.radar_fields) if radar else None, timestamp / 1e9)
        self.stats['processed'] += 1
        annotated = self.processor.annotate(image, result)
        preview_width = int(self.processor.cfg.get('visualization', {}).get('preview_width', 960))
        if preview_width > 0 and annotated.shape[1] > preview_width:
            preview_height = round(annotated.shape[0] * preview_width / annotated.shape[1])
            annotated = cv2.resize(annotated, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
        output = Image(header=msg.header, height=annotated.shape[0], width=annotated.shape[1],
                       encoding='bgr8', is_bigendian=0, step=annotated.shape[1] * 3,
                       data=array('B', annotated.tobytes()))
        self.image_pub.publish(output)
        header = Header(stamp=msg.header.stamp, frame_id='velodyne')
        self.cloud_pub.publish(xyz_cloud(header, result['nonground']))
        points, identifiers = instance_points(result['nonground'], result['tracks'])
        self.instance_pub.publish(instance_cloud(header, points, identifiers))
        self.publish_objects(header, result['tracks'])
        self.proposal_pub.publish(self.scene.proposals(header, result.get('lidar_proposals', [])))
        metrics = dict(result['metrics'], **self.stats)
        metrics['lidar_delta_ms'] = (stamp_ns(lidar) - timestamp) / 1e6
        metrics['radar_delta_ms'] = (stamp_ns(radar) - timestamp) / 1e6 if radar else None
        metrics['node_latency_ms'] = (time.perf_counter() - node_started) * 1000
        self.metrics_pub.publish(String(data=json.dumps(metrics)))
        if self.stats['processed'] % 25 == 0:
            self.get_logger().info(f"Processed {self.stats['processed']} frames; "
                f"tracks={len(result['tracks'])}, latency={metrics['latency_ms']:.1f}ms")

    def publish_objects(self, header, tracks):
        scene = getattr(self, 'scene', None)
        if scene is None:
            scene = SceneRenderer(self.processor.cfg.get('visualization', {}))
        detections = Detection3DArray(header=header)
        track_data = []
        for track in tracks:
            obj = track.object
            x, y = map(float, track.state[:2])
            z = float(obj.center[2])
            detection = Detection3D(header=header, id=str(track.id))
            pose = detection.bbox.center
            pose.position.x, pose.position.y, pose.position.z = x, y, z
            yaw = getattr(obj, 'yaw', 0.)
            pose.orientation.z = float(np.sin(yaw / 2))
            pose.orientation.w = float(np.cos(yaw / 2))
            detection.bbox.size.x, detection.bbox.size.y, detection.bbox.size.z = map(float, obj.size)
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = obj.label
            hypothesis.hypothesis.score = float(obj.confidence)
            hypothesis.pose.pose = pose
            hypothesis.pose.covariance[14] = 1.0  # Heuristic z variance; not tracked.
            for axis in (3, 4, 5):
                hypothesis.pose.covariance[axis * 6 + axis] = 1e3  # Orientation is not estimated.
            for r in range(2):
                for c in range(2):
                    hypothesis.pose.covariance[r * 6 + c] = float(track.covariance[r, c])
            detection.results = [hypothesis]
            detections.detections.append(detection)
            track_data.append({'id': track.id, 'position': [x, y, z],
                'velocity_relative_xy': track.state[2:].tolist(),
                'state_covariance': track.covariance.tolist(), 'label': obj.label,
                'sources': obj.sources, 'measured': track.measured,
                'radial_velocity_sensor': obj.radial_velocity})
        self.marker_pub.publish(scene.tracks(header, tracks))
        if hasattr(self, 'box_pub'):
            self.box_pub.publish(scene.tracks(header, tracks, layer='boxes'))
            self.motion_pub.publish(scene.tracks(header, tracks, layer='motion'))
        self.detection_pub.publish(detections)
        self.track_pub.publish(String(data=json.dumps({'frame_id': header.frame_id,
            'timestamp_ns': header.stamp.sec * 10**9 + header.stamp.nanosec, 'tracks': track_data})))


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
