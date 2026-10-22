"""Geometry and camera-display checks independent of ROS or model inference."""

from types import SimpleNamespace
import numpy as np
from group8_fusion.geometry import oriented_corners, BOX_EDGES
from group8_fusion.display import (
    draw_projected_box,
    annotate_frame,
    track_bgr,
    track_rgb,
)


def test_oriented_wireframe_preserves_box_edges_at_ninety_degrees():
    corners = oriented_corners([10, 2, 0], [4, 2, 1.5], np.pi / 2)
    assert np.allclose(np.ptp(corners, axis=0), [2, 4, 1.5])
    assert np.allclose(corners.mean(0), [10, 2, 0])
    lengths = [np.linalg.norm(corners[a] - corners[b]) for a, b in BOX_EDGES]
    assert np.allclose(sorted(lengths), sorted([4.0, 2.0, 1.5] * 4))


def test_camera_box_crossing_near_plane_is_not_projected():
    image = np.zeros((100, 160, 3), np.uint8)
    draw_projected_box(
        image, [0, 0, 0.5], [2, 2, 2], 0, np.eye(4), np.eye(3), (0, 255, 0)
    )
    assert not image.any()


def test_identity_colors_agree_between_camera_bgr_and_rviz_rgb():
    for identifier in range(1, 9):
        assert np.allclose(
            np.array(track_bgr(identifier))[::-1] / 255, track_rgb(identifier)
        )
    assert len({track_rgb(i) for i in range(1, 9)}) == 8


def test_camera_rendering_preserves_input_and_does_not_invent_detections():
    image = np.full((240, 400, 3), 60, np.uint8)
    original = image.copy()
    processor = SimpleNamespace(
        cfg={"camera": {"status": "approximate"}},
        camera_from_lidar=np.eye(4),
        K=np.eye(3),
    )
    result = dict(
        tracks=[],
        boxes=[],
        nonground=np.empty((0, 3)),
        metrics={
            "tracks": 0,
            "radar_matches": 0,
            "camera_detections": 0,
            "lidar_objects": 0,
            "latency_ms": 0.0,
        },
    )
    rendered = annotate_frame(processor, image, result)
    assert np.array_equal(image, original)
    assert not np.array_equal(rendered, original)
    assert not result["tracks"] and not result["boxes"]
