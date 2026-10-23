"""Metric alignment and support contracts for dense middle-fusion features."""

from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from group8_fusion.mid_features import (
    CameraSpatialEncoder, LidarSpatialEncoder, camera_bev, centers, lidar_bev,
    radar_bev,
)


def grid(**changes):
    return dict(dict(x_min=0., y_min=-24., cell_size=.75, width=4, height=3), **changes)


def test_grid_centres_use_columns_for_x_rows_for_y():
    points = centers(grid(), "cpu")
    assert points.shape == (3, 4, 2)
    torch.testing.assert_close(points[0, 0], torch.tensor([.375, -23.625]))
    torch.testing.assert_close(points[2, 3], torch.tensor([2.625, -22.125]))


def test_lidar_sparse_xy_alignment_and_local_interior_support():
    values = torch.ones(1, 128, requires_grad=True)
    sparse = dict(features=values, indices=torch.tensor([[0, 4, 4]]), shape=(9, 9))
    # VoxelNeXt regression uses -54 + .6 * index as the index reference.
    metric = grid(x_min=-54.3, y_min=-54.3, cell_size=.6, width=9, height=9)
    features, support = lidar_bev(sparse, metric)
    torch.testing.assert_close(features[:, 4, 4], torch.ones(128), atol=2e-5, rtol=0)
    assert support[0, 3, 3] > .99  # Interior may be one cell away from returns.
    assert support[0, 2, 2] > .99  # Two-cell local context is explicitly supported.
    assert support[0, 0, 0] == 0
    assert features[:, 3, 3].abs().max() < 2e-5  # Features are never dilated.
    features[:, 4, 4].sum().backward()
    assert torch.isfinite(values.grad).all() and values.grad.sum() > 0


def test_lidar_empty_and_outside_source_grid_have_no_invented_support():
    sparse = dict(features=torch.empty(0, 128), indices=torch.empty(0, 3, dtype=torch.long), shape=(9, 9))
    features, support = lidar_bev(sparse, grid())
    assert features.shape == (128, 3, 4) and support.shape == (1, 3, 4)
    assert not features.any() and not support.any()
    occupied = dict(features=torch.ones(1, 128), indices=torch.tensor([[0, 4, 4]]), shape=(9, 9))
    features, support = lidar_bev(occupied, grid())
    assert not features.any() and not support.any()


def test_camera_alignment_includes_centered_yolo_letterbox_and_height_support():
    # A feature ramp with value u/640 + 2*v/640 in the padded input frame.
    coordinate = (torch.arange(64, dtype=torch.float32) + .5) / 64
    image_features = (coordinate[None] + 2 * coordinate[:, None])[None, None].requires_grad_()
    metric = grid(x_min=0., y_min=0., cell_size=1., width=2, height=2)
    features, support = camera_bev(
        image_features, metric, np.eye(4), np.eye(3), width=4, height=4
    )
    # Only z=1 is in front of this synthetic identity camera. For x=y=.5,
    # padded pixel is (80,170), giving .125 + 2*.265625 = .65625.
    torch.testing.assert_close(features[0, 0, 0], torch.tensor(.65625))
    torch.testing.assert_close(support, torch.full((1, 2, 2), 1 / 3))
    features.sum().backward()
    assert image_features.grad.abs().sum() > 0 and torch.isfinite(image_features.grad).all()


def test_camera_behind_the_rig_has_zero_support_and_features():
    transform = np.eye(4)
    transform[2, 3] = -10
    feature, support = camera_bev(torch.ones(1, 192, 8, 8), grid(), transform, np.eye(3))
    assert not support.any() and not feature.any()
    assert torch.isfinite(feature).all()


def test_camera_tensor_preprocessing_is_rgb_with_exact_vertical_padding():
    encoder = CameraSpatialEncoder.__new__(CameraSpatialEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.device = "cpu"
    image = np.full((1200, 1920, 3), [10, 20, 30], np.uint8)
    result = encoder.tensor(image)
    assert result.shape == (1, 3, 640, 640)
    torch.testing.assert_close(result[0, :, 120, 0], torch.tensor([30., 20., 10.]) / 255)
    torch.testing.assert_close(result[0, :, :120], torch.full((3, 120, 640), 114 / 255))
    torch.testing.assert_close(result[0, :, 520:], torch.full((3, 120, 640), 114 / 255))


def test_radar_attributes_offsets_count_and_height_variance_are_explicit():
    metric = grid(x_min=0., y_min=0., cell_size=1., width=3, height=3)
    raw = np.array([[1.25, 2.25, 1.5, 15., 25.], [1.75, 2.75, -1.5, -15., -25.]], np.float32)
    features, support = radar_bev(raw, np.eye(4), metric, device="cpu")
    assert features.shape == (8, 3, 3) and support.sum() == 1
    expected = torch.tensor([
        np.log(3) / 4, 0., 0., 0., 0., 0., .25,
        (np.hypot(1.25, 2.25) + np.hypot(1.75, 2.75)) / (2 * 65),
    ], dtype=torch.float32)
    torch.testing.assert_close(features[:, 2, 1], expected)
    assert not features[:, :2].any()


def test_radar_extrinsic_translation_and_upper_boundary_are_respected():
    metric = grid(x_min=0., y_min=0., cell_size=1., width=3, height=3)
    transform = np.eye(4)
    transform[0, 3] = 1
    raw = np.array([[.5, .5, 0, 0, 0], [2., .5, 0, 0, 0]], np.float32)
    _, support = radar_bev(raw, transform, metric, device="cpu")
    assert support[0, 0, 1] == 1 and support.sum() == 1  # x=3 is outside.


def test_radar_empty_nonfinite_and_out_of_range_returns_are_safe():
    metric = grid(x_min=0., y_min=0., cell_size=1., width=3, height=3)
    feature, support = radar_bev(np.empty((0, 5), np.float32), np.eye(4), metric, device="cpu")
    assert not feature.any() and not support.any()
    raw = np.array([
        [np.nan, 0, 0, 0, 0], [1, np.inf, 0, 0, 0], [-1, 1, 0, 0, 0],
        [1, 1, 3, 0, 0], [1.5, 1.5, 0, np.nan, np.inf],
    ], np.float32)
    feature, support = radar_bev(raw, np.eye(4), metric, device="cpu")
    assert torch.isfinite(feature).all() and support.sum() == 1
    assert feature[4, 1, 1] == 0 and feature[5, 1, 1] == 3


def test_lidar_spatial_encoder_can_share_an_existing_published_backbone():
    class Detector:
        model = torch.nn.Linear(4, 128)

    existing = Detector()
    encoder = LidarSpatialEncoder({}, detector=existing)
    assert encoder.detector is existing and encoder.network is existing.model
    assert all(not p.requires_grad for p in encoder.network.parameters())


@pytest.mark.parametrize("raw", [
    np.empty((0, 4), np.float32),
    np.array([[np.nan, 0, 0, 1], [1, 1, 0, np.inf]], np.float32),
    np.array([[55, 0, 0, 1], [0, -55, 0, 1], [0, 0, -6, 1], [54, 0, 0, 1]], np.float32),
])
@pytest.mark.parametrize("teacher", [False, True])
def test_empty_nonfinite_and_outside_lidar_avoid_voxel_and_sparse_backends(raw, teacher):
    class Dataset:
        point_cloud_range = np.array([-54., -54., -5., 54., 54., 3.])

        def prepare_data(self, data):
            raise AssertionError("Empty sensor input reached voxelization")

    class Detector:
        model = torch.nn.Linear(4, 128)
        dataset = Dataset()

    encoder = LidarSpatialEncoder({}, detector=Detector())
    result = encoder(raw, teacher=teacher)
    assert result["features"].shape == (0, 128)
    assert result["features"].device == next(encoder.network.parameters()).device
    assert result["indices"].shape == (0, 3) and result["indices"].dtype == torch.int32
    assert result["shape"] == (180, 180)
    if teacher:
        assert result["teacher_boxes"].shape == (0, 9)
        assert result["teacher_labels"].dtype == np.int64
        assert result["teacher_scores"].dtype == np.float32
    else:
        assert "teacher_boxes" not in result


def test_post_voxelization_empty_lidar_returns_without_collation_or_gpu_transfer():
    class Dataset:
        def prepare_data(self, data):
            np.testing.assert_allclose(data["points"], [[1, 2, 0, 1, 0]])
            return dict(voxels=np.empty((0, 10, 5)), voxel_coords=np.empty((0, 3)))

        def collate_batch(self, data):
            raise AssertionError("Empty voxel set reached collation")

    class Detector:
        model = torch.nn.Linear(4, 128)
        dataset = Dataset()
        intensity_scale = 255.

    result = LidarSpatialEncoder({}, detector=Detector())(
        np.array([[1., 2., 0., 255.]], np.float32), teacher=True
    )
    assert result["features"].shape == (0, 128) and result["teacher_boxes"].shape == (0, 9)


@pytest.mark.skipif(not Path("/models/yolo11n.pt").is_file(), reason="Published YOLO weights are mounted only in the project GPU container")
def test_published_yolo_feature_taps_are_spatial_and_differentiable():
    pytest.importorskip("ultralytics")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = CameraSpatialEncoder(device=device)
    image = torch.rand(1, 3, 640, 640, device=device, requires_grad=True)
    features = encoder(image)
    assert features.shape == (1, 192, 80, 80)
    features.square().mean().backward()
    assert image.grad is not None and image.grad.abs().sum() > 0
    assert torch.isfinite(image.grad).all()
