"""Checkpoint compatibility and independence of spatial detections from camera IDs."""

import hashlib
from types import SimpleNamespace
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from group8_fusion import middle
from group8_fusion.mid_features import GRID, NAMES
from group8_fusion.mid_model import BEVMiddleFusion
from group8_fusion.mid_training import preprocessing_contract


@pytest.fixture
def checkpoint(tmp_path):
    camera, lidar = tmp_path / "camera.pt", tmp_path / "lidar.pth"
    camera.write_bytes(b"fixed camera encoder checkpoint")
    lidar.write_bytes(b"fixed lidar encoder checkpoint")
    intrinsics = [1100., 0., 960., 0., 1100., 600., 0., 0., 1.]
    cfg = dict(camera=dict(K=intrinsics, width=1920, height=1200),
               middle=dict(camera_weights=str(camera)),
               neural=dict(lidar=dict(weights=str(lidar), intensity_scale=255.)))
    data = dict(schema="group8-middle-fusion-v1", experiment="middle_fusion",
                supervision="teacher_distillation", grid=GRID.copy(),
                feature_contract=preprocessing_contract(cfg),
                camera_intrinsics=intrinsics, score_threshold=.35,
                architecture=dict(camera_channels=192, lidar_channels=128,
                                  radar_channels=8, dimension=32, classes=10),
                model_hashes={"yolo11n.pt": hashlib.sha256(camera.read_bytes()).hexdigest(),
                              "voxelnext_nuscenes.pth": hashlib.sha256(lidar.read_bytes()).hexdigest()},
                state_dict=BEVMiddleFusion().state_dict())
    path = tmp_path / "middle.pt"
    return path, cfg, data


def save_and_load(checkpoint):
    path, cfg, data = checkpoint
    torch.save(data, path)
    return middle.load_middle_checkpoint(path, cfg)


def test_safe_checkpoint_load_returns_eval_model_exact_threshold_and_file_hash(checkpoint):
    model, threshold, digest = save_and_load(checkpoint)
    assert not model.training and threshold == .35
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    assert digest == hashlib.sha256(checkpoint[0].read_bytes()).hexdigest()


def test_missing_checkpoint_fails_before_constructing_any_encoder(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train the spatial detector"):
        middle.load_middle_checkpoint(tmp_path / "absent.pt", {})


@pytest.mark.parametrize("field,value", [("schema", "group8-query-fusion-v2"),
                                         ("experiment", "lidar_only")])
def test_incompatible_heads_are_rejected(checkpoint, field, value):
    checkpoint[2][field] = value
    with pytest.raises(ValueError, match="trained middle fusion checkpoint"):
        save_and_load(checkpoint)


def test_wrong_supervision_cannot_be_presented_as_teacher_distillation(checkpoint):
    checkpoint[2]["supervision"] = "ground_truth"
    with pytest.raises(ValueError, match="teacher-distillation supervision"):
        save_and_load(checkpoint)


@pytest.mark.parametrize("name,value", [("dimension", 64), ("classes", 11),
                                        ("radar_channels", 7), ("dimension", 32.)])
def test_unexpected_architecture_fails_before_model_construction(checkpoint, name, value):
    checkpoint[2]["architecture"][name] = value
    with pytest.raises(ValueError, match="architecture differs"):
        save_and_load(checkpoint)


@pytest.mark.parametrize("name,value", [("camera_width", 1280), ("camera_height", 720),
                                        ("intensity_scale", 1.),
                                        ("camera_lift_heights", [-1., 0., 2.]),
                                        ("lidar_support_radius_cells", 1)])
def test_preprocessing_changes_require_reextraction(checkpoint, name, value):
    checkpoint[2]["feature_contract"][name] = value
    with pytest.raises(ValueError, match="preprocessing changed"):
        save_and_load(checkpoint)


def test_grid_or_intrinsic_changes_require_reextraction(checkpoint):
    checkpoint[2]["grid"]["cell_size"] = 1.
    with pytest.raises(ValueError, match="grid changed"):
        save_and_load(checkpoint)
    checkpoint[2]["grid"] = GRID.copy()
    checkpoint[1]["camera"]["K"] = [1200., 0., 960., 0., 1100., 600., 0., 0., 1.]
    with pytest.raises(ValueError, match="intrinsics changed"):
        save_and_load(checkpoint)


@pytest.mark.parametrize("name", ["yolo11n.pt", "voxelnext_nuscenes.pth"])
def test_encoder_hash_changes_are_rejected(checkpoint, name):
    checkpoint[2]["model_hashes"][name] = "wrong checkpoint hash"
    with pytest.raises(ValueError, match="encoder checkpoint changed"):
        save_and_load(checkpoint)


@pytest.mark.parametrize("threshold", [0., -1., 1.1, float("nan"), float("inf"), None, "invalid"])
def test_invalid_calibrated_threshold_is_rejected(checkpoint, threshold):
    checkpoint[2]["score_threshold"] = threshold
    with pytest.raises(ValueError, match="Invalid calibrated detector threshold"):
        save_and_load(checkpoint)


def test_missing_or_nonfinite_learned_parameters_are_rejected(checkpoint):
    parameter = next(iter(checkpoint[2]["state_dict"]))
    saved = checkpoint[2]["state_dict"].pop(parameter)
    with pytest.raises(RuntimeError, match="Missing key"):
        save_and_load(checkpoint)
    checkpoint[2]["state_dict"][parameter] = saved
    checkpoint[2]["state_dict"][parameter].flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite trained middle fusion parameters"):
        save_and_load(checkpoint)


def test_independent_detections_survive_when_no_camera_identity_is_available(monkeypatch):
    processor = middle.MiddleProcessor.__new__(middle.MiddleProcessor)
    processor.cfg = dict(camera=dict(width=8, height=8), middle=dict(max_detections=64))
    processor.last_time = None
    processor.middle_threshold = .5
    processor.fusion_schema = "group8-query-fusion-v2"
    processor.quality_rejections = 0
    processor.camera_from_lidar = np.eye(4)
    processor.lidar_from_radar = np.eye(4)
    processor.K = np.eye(3)
    processor.detector = SimpleNamespace(detect=lambda image: [])
    processor.visual_tracker = SimpleNamespace(update=lambda image, boxes: [],
                                              capacity_rejections=0)
    processor.lidar_detector = SimpleNamespace(names=NAMES)
    calls = []

    def frozen_lidar(raw, teacher):
        calls.append(teacher)
        if teacher:
            raise AssertionError("Published final detection head must not run at inference")
        return dict(features=torch.empty(0, 128), indices=torch.empty(0, 3, dtype=torch.long))

    class FrozenCamera:
        def tensor(self, image):
            return torch.zeros(1, 3, 8, 8)

        def __call__(self, tensor):
            return torch.zeros(1, 192, 8, 8)

    def independent_head(**inputs):
        assert set(inputs) == {"camera", "lidar", "radar", "availability"}
        logits = torch.full((1, 10, 64, 64), -20.)
        logits[0, 0, 32, 12] = 5
        regression = torch.zeros(1, 8, 64, 64)
        regression[:, 7] = 1
        return dict(heatmap_logits=logits, box_regression=regression,
                    gates=torch.ones(1, 3, 64, 64) / 3)

    processor.spatial_lidar = frozen_lidar
    processor.spatial_camera = FrozenCamera()
    processor.middle_network = independent_head
    processor.infer_record = lambda record: ([], {})
    monkeypatch.setattr(middle, "aligned_features", lambda *args: {
        name: torch.zeros(channels, 64, 64)
        for name, channels in (("camera", 192), ("lidar", 128), ("radar", 8), ("availability", 3))
    })
    result = processor.process(np.zeros((8, 8, 3), np.uint8),
                               np.empty((0, 4), np.float32), None, 1.)
    assert calls == [False]
    assert not result["tracks"] and not result["objects"]
    assert result["metrics"]["detector_objects"] == 1
    assert len(result["detector_objects"]) == 1
    assert result["detector_objects"][0]["origin"] == "BEV fusion"
    assert result["detector_objects"][0]["label"] == "car"
