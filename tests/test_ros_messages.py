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
from group8_fusion.ros_node import FusionNode


def test_humble_track_outputs_serialize():
    adapter = SimpleNamespace(marker_pub=Mock(), detection_pub=Mock(), track_pub=Mock())
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
