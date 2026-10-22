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


def test_oriented_scene_serializes_and_preserves_real_box_geometry():
    from visualization_msgs.msg import Marker
    from group8_fusion.scene import SceneRenderer
    from group8_fusion.geometry import oriented_corners

    obj = Object3D(np.array([10., 1., 0.]), np.array([4., 2., 1.5]), 'car', .9, ['camera', 'lidar'])
    obj.yaw = np.pi / 2
    track = Track(4, np.array([10., 1., 100., 0.]), np.eye(4), obj, 0., 2)
    renderer = SceneRenderer({'range_rings': False})
    message = renderer.tracks(Header(frame_id='velodyne'), [track], layer='boxes')
    assert serialize_message(message)
    outline = next(m for m in message.markers if m.ns == 'box_outlines')
    assert len(outline.points) == 24
    points = np.array([[p.x, p.y, p.z] for p in outline.points])
    expected = oriented_corners(obj.center, obj.size, obj.yaw)
    assert {tuple(p) for p in points} == {tuple(p) for p in expected}
    motion = renderer.tracks(Header(frame_id='velodyne'), [track], layer='motion')
    arrow = next(m for m in motion.markers if m.ns == 'relative_velocity')
    a, b = arrow.points
    assert np.isclose(np.linalg.norm([b.x-a.x, b.y-a.y, b.z-a.z]), 4.)
    # Disappearance and rewind cannot leave old object markers behind.
    empty = renderer.tracks(Header(frame_id='velodyne'), [])
    assert len(empty.markers) == 1 and empty.markers[0].action == Marker.DELETEALL


def test_unfused_proposals_are_a_separate_layer_and_invalid_boxes_are_ignored():
    from group8_fusion.scene import SceneRenderer

    proposals = [dict(box=np.array([10., 0, 0, 4, 2, 1.5, .2]), label='car'),
                 dict(box=np.array([np.nan, 0, 0, 4, 2, 1.5, 0]), label='invalid')]
    message = SceneRenderer().proposals(Header(frame_id='velodyne'), proposals)
    assert serialize_message(message)
    assert len(message.markers) == 3
    assert all(m.ns.startswith('unfused_') for m in message.markers[1:])
    assert 'LiDAR only' in message.markers[-1].text


def test_colored_instance_cloud_preserves_real_points_and_identity_colors():
    from group8_fusion.ros_node import instance_cloud
    from group8_fusion.display import PALETTE

    points = np.array([[10., 1., .5], [11., 2., .8]], np.float32)
    message = instance_cloud(Header(frame_id='velodyne'), points, [1, 2])
    restored = deserialize_message(serialize_message(message), PointCloud2)
    assert np.allclose(cloud_array(restored), points)
    assert restored.point_step == 16 and restored.header.frame_id == 'velodyne'
    color = np.frombuffer(restored.data, dtype='<u4').reshape(-1, 4)[:, 3]
    assert [(int(c) >> 16, (int(c) >> 8) & 255, int(c) & 255) for c in color] == list(PALETTE[:2])


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
