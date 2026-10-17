"""Check actual Humble message construction; CI without ROS skips this module."""
from types import SimpleNamespace
from unittest.mock import Mock
import json
import numpy as np
import pytest

pytest.importorskip('rclpy')
from rclpy.serialization import serialize_message
from std_msgs.msg import Header
from group8_fusion.perception import Object3D
from group8_fusion.tracking import Track
from group8_fusion.ros_node import FusionNode, xyz_cloud
from group8_fusion.geometry import cloud_array
from sensor_msgs.msg import PointCloud2, Image
from group8_fusion.bag import restamp_sensor_cdr
from rclpy.serialization import deserialize_message


def test_humble_track_outputs_serialize():
    adapter = SimpleNamespace(marker_pub=Mock(), detection_pub=Mock(), track_pub=Mock(), processor=SimpleNamespace(cfg={}))
    obj = Object3D(np.array([10., 1., 0.]), np.array([4., 2., 1.5]), 'car', .9, ['camera', 'lidar'])
    track = Track(4, np.array([10., 1., .2, 0.]), np.eye(4), obj, 0., 2)
    header = Header(frame_id='velodyne')
    header.stamp.sec = 100
    FusionNode.publish_objects(adapter, header, [track])
    for publisher in (adapter.marker_pub, adapter.detection_pub, adapter.track_pub):
        assert serialize_message(publisher.publish.call_args.args[0])
    msg = adapter.detection_pub.publish.call_args.args[0]
    assert msg.detections[0].id == '4'
    assert msg.detections[0].results[0].hypothesis.class_id == 'car'
    state = json.loads(adapter.track_pub.publish.call_args.args[0].data)
    assert state['tracks'][0]['sources'] == ['camera', 'lidar']


def test_cloud_cdr_roundtrip_preserves_coordinates_and_frame():
    points = np.array([[1.25, -2.5, 3.], [4., 5., 6.]], dtype=np.float32)
    header = Header(frame_id='velodyne')
    cloud = xyz_cloud(header, points)
    restored = deserialize_message(serialize_message(cloud), PointCloud2)
    assert restored.header.frame_id == 'velodyne'
    assert restored.row_step == 24
    assert np.allclose(cloud_array(restored), points)


@pytest.mark.parametrize('message', [Image(header=Header(frame_id='cam0'), width=2, height=1,
    encoding='bgr8', step=6, data=[1, 2, 3, 4, 5, 6]),
    xyz_cloud(Header(frame_id='velodyne'), np.array([[1., 2., 3.]], dtype=np.float32))])
def test_serialized_replay_restamps_without_changing_payload(message):
    original = serialize_message(message)
    restamped = restamp_sensor_cdr(original, 12345678901234)
    restored = deserialize_message(restamped, type(message))
    assert restored.header.stamp.sec == 12345
    assert restored.header.stamp.nanosec == 678901234
    assert restored.header.frame_id == message.header.frame_id
    assert original[12:] == restamped[12:]
    assert restored.data == message.data
