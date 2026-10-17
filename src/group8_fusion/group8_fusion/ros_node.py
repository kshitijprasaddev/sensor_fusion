"""ROS adapter with bounded synchronization queues and RViz outputs."""
from collections import deque
import json
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from .bag import Bag, CAMERA, LIDAR, RADAR
from .geometry import cloud_array, image_array
from .processor import load_config, CameraDetector, Processor


def stamp_ns(msg):
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


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
        self.processor = Processor(cfg, bag.lidar_from('cam0'), bag.lidar_from('radar'), CameraDetector(cfg['model']))
        bag.close()
        self.tolerance = int(cfg['synchronization']['tolerance_ms'] * 1e6)
        self.clouds = {LIDAR: deque(maxlen=40), RADAR: deque(maxlen=60)}
        self.images = deque(maxlen=4)
        self.last_stamp = None
        self.stats = {'processed': 0, 'dropped_no_lidar': 0, 'dropped_images': 0}
        self.image_pub = self.create_publisher(Image, '/fusion/image', 2)
        self.cloud_pub = self.create_publisher(PointCloud2, '/fusion/nonground', 2)
        self.marker_pub = self.create_publisher(MarkerArray, '/fusion/markers', 2)
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
            self.get_logger().warning('APPROXIMATE camera intrinsics. Projection is a demonstration until calibrated.')

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
            self.processor.tracker.tracks.clear()
            self.processor.tracker.last_time = None
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
        if time.monotonic() - received < 0.5 and (latest is None or stamp_ns(latest) < timestamp + self.tolerance):
            return
        self.images.popleft()
        lidar, radar = self.nearest(LIDAR, timestamp), self.nearest(RADAR, timestamp)
        if lidar is None:
            self.stats['dropped_no_lidar'] += 1
            return
        image = image_array(msg)
        result = self.processor.process(image, cloud_array(lidar),
            cloud_array(radar, ('x', 'y', 'z', 'v')) if radar else None, timestamp / 1e9)
        self.stats['processed'] += 1
        annotated = self.processor.annotate(image, result)
        output = Image(header=msg.header, height=annotated.shape[0], width=annotated.shape[1],
                       encoding='bgr8', is_bigendian=0, step=annotated.shape[1] * 3,
                       data=annotated.tobytes())
        self.image_pub.publish(output)
        header = Header(stamp=msg.header.stamp, frame_id='velodyne')
        self.cloud_pub.publish(point_cloud2.create_cloud_xyz32(header, result['nonground'].astype(np.float32)))
        self.publish_objects(header, result['tracks'])
        metrics = dict(result['metrics'], **self.stats)
        metrics['lidar_delta_ms'] = (stamp_ns(lidar) - timestamp) / 1e6
        metrics['radar_delta_ms'] = (stamp_ns(radar) - timestamp) / 1e6 if radar else None
        self.metrics_pub.publish(String(data=json.dumps(metrics)))
        if self.stats['processed'] % 25 == 0:
            self.get_logger().info(f"Processed {self.stats['processed']} frames; "
                f"tracks={len(result['tracks'])}, latency={metrics['latency_ms']:.1f}ms")

    def publish_objects(self, header, tracks):
        clear = Marker(action=Marker.DELETEALL)
        markers = [clear]
        detections = Detection3DArray(header=header)
        track_data = []
        for track in tracks:
            obj = track.object
            x, y = map(float, track.state[:2])
            z = float(obj.center[2])
            color = (0.25, 1., 0.5) if len(obj.sources) >= 2 else (0.2, 0.65, 1.)
            cube = Marker(header=header, ns='tracks', id=track.id, type=Marker.CUBE, action=Marker.ADD)
            cube.pose.position.x, cube.pose.position.y, cube.pose.position.z = x, y, z
            cube.pose.orientation.w = 1.
            cube.scale.x, cube.scale.y, cube.scale.z = map(float, obj.size)
            cube.color.r, cube.color.g, cube.color.b = color
            cube.color.a = 0.4 if track.measured else 0.15
            cube.lifetime.sec = 1
            markers.append(cube)
            text = Marker(header=header, ns='labels', id=track.id, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
            text.pose.position.x, text.pose.position.y, text.pose.position.z = x, y, z + float(obj.size[2]) / 2 + 0.5
            text.pose.orientation.w = 1.
            text.scale.z = 0.45
            text.color.r = text.color.g = text.color.b = text.color.a = 1.
            text.text = f'#{track.id} {obj.label}\n{"+".join(obj.sources)} | v_rel {np.linalg.norm(track.state[2:]):.1f}m/s'
            text.lifetime.sec = 1
            markers.append(text)
            arrow = Marker(header=header, ns='velocity', id=track.id, type=Marker.ARROW, action=Marker.ADD)
            arrow.points = [Point(x=x, y=y, z=z), Point(x=x + float(track.state[2]), y=y + float(track.state[3]), z=z)]
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.08, 0.16, 0.2
            arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 1., 0.7, 0.2, 1.
            arrow.lifetime.sec = 1
            markers.append(arrow)
            detection = Detection3D(header=header, id=str(track.id))
            detection.bbox.center = cube.pose
            detection.bbox.size = cube.scale
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = obj.label
            hypothesis.hypothesis.score = float(obj.confidence)
            hypothesis.pose.pose = cube.pose
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
        self.marker_pub.publish(MarkerArray(markers=markers))
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
