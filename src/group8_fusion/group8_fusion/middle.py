"""Independent spatial fusion detection, with the established visual identity adapter."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from .neural import NeuralProcessor
from .attention_fusion import pack_features
from .mid_features import CameraSpatialEncoder, LidarSpatialEncoder, aligned_features, GRID
from .mid_model import BEVMiddleFusion, decode
from .mid_training import preprocessing_contract


ARCHITECTURE = dict(camera_channels=192, lidar_channels=128, radar_channels=8,
                    dimension=32, classes=10)


def load_middle_checkpoint(path, cfg):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Train the spatial detector first: ./scripts/run.sh mid-extract then mid-train')
    data = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(data, dict) or data.get('schema') != 'group8-middle-fusion-v1' or data.get('experiment') != 'middle_fusion':
        raise ValueError('Expected the trained middle fusion checkpoint, not an association or LiDAR-only head')
    if data.get('supervision') != 'teacher_distillation':
        raise ValueError('Expected disclosed teacher-distillation supervision for the spatial detector')
    architecture = data.get('architecture')
    if not isinstance(architecture, dict) or architecture != ARCHITECTURE or any(type(v) is not int for v in architecture.values()):
        raise ValueError('Spatial detector architecture differs from the pinned feature and class schema')
    if data.get('grid') != GRID:
        raise ValueError('BEV grid changed: re-extract spatial features and retrain')
    try:
        intrinsics = np.asarray(data.get('camera_intrinsics', []), dtype=float)
    except (ValueError, TypeError):
        intrinsics = np.empty(0)
    if intrinsics.shape != (9,) or not np.isfinite(intrinsics).all() or not np.allclose(intrinsics, cfg['camera']['K']):
        raise ValueError('Camera intrinsics changed: re-extract spatial features and retrain')
    if data.get('feature_contract') != preprocessing_contract(cfg):
        raise ValueError('Spatial preprocessing changed: re-extract spatial features and retrain')
    for name, filename in [('yolo11n.pt', cfg['middle']['camera_weights']),
                           ('voxelnext_nuscenes.pth', cfg['neural']['lidar']['weights'])]:
        if data.get('model_hashes', {}).get(name) != hashlib.sha256(Path(filename).read_bytes()).hexdigest():
            raise ValueError(f'Spatial encoder checkpoint changed: {name}; re-extract and retrain')
    try:
        threshold = float(data['score_threshold'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Invalid calibrated detector threshold') from error
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('Invalid calibrated detector threshold')
    model = BEVMiddleFusion(**data['architecture'])
    model.load_state_dict(data['state_dict'], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Non-finite trained middle fusion parameters')
    return model.eval(), threshold, hashlib.sha256(path.read_bytes()).hexdigest()


class MiddleProcessor(NeuralProcessor):
    """Fuse complete sensor maps, regress boxes, then attach optional camera identities.

    The inherited association head was trained on VoxelNeXt candidates. Its use on
    new detector candidates is an explicitly experimental transfer, not retraining.
    All independent detections remain available even without accepted camera IDs.
    """
    def __init__(self, cfg, lidar_from_camera, lidar_from_radar, detector=None):
        model, self.middle_threshold, self.middle_checkpoint_sha256 = load_middle_checkpoint(cfg['middle']['weights'], cfg)
        super().__init__(cfg, lidar_from_camera, lidar_from_radar, detector)
        self.spatial_camera = CameraSpatialEncoder(cfg['middle']['camera_weights'])
        self.spatial_lidar = LidarSpatialEncoder(cfg['neural']['lidar'], detector=self.lidar_detector)
        self.middle_network = model.cuda().eval()
        self.middle_gate_mean = [0., 0., 0.]

    def extract(self, image, lidar, radar, timestamp):
        if self.last_time is not None and (timestamp <= self.last_time or timestamp - self.last_time > 2):
            self.reset_tracking()
        self.last_time = timestamp
        if image.shape[1::-1] != (self.cfg['camera']['width'], self.cfg['camera']['height']):
            raise ValueError('Image dimensions differ from camera calibration')
        boxes = self.detector.detect(image)
        visual = self.visual_tracker.update(image, boxes)
        raw = np.empty((0, 5), np.float32) if radar is None else radar
        with torch.inference_mode():
            # teacher=False is essential: the published final detection head is
            # never executed on the middle inference path.
            sparse = self.spatial_lidar(lidar, teacher=False)
            features = aligned_features(self.spatial_camera(self.spatial_camera.tensor(image)),
                                        sparse, raw, GRID, self.camera_from_lidar,
                                        self.K, self.lidar_from_radar, image.shape[1], image.shape[0])
            prediction = self.middle_network(**{key: value[None] for key, value in features.items()})
            detections = decode(prediction, GRID, threshold=self.middle_threshold,
                                max_detections=self.cfg['middle']['max_detections'])
            self.middle_gate_mean = prediction['gates'].mean((0, 2, 3)).cpu().tolist()
            if len(detections['boxes']) and len(sparse['features']):
                xy = sparse['indices'][:, [2, 1]].float() * .6 - 54
                nearest = torch.cdist(torch.as_tensor(detections['boxes'][:, :2], device='cuda'), xy).argmin(1)
                encoded = sparse['features'][nearest].float().cpu().numpy()
            else:
                encoded = np.zeros((len(detections['boxes']), 128), np.float32)
        detections.update(features=encoded, names=self.lidar_detector.names)
        record = pack_features(visual, detections, raw, self.lidar_from_radar,
                               self.camera_from_lidar, self.K, image.shape[1], image.shape[0])
        record['time'] = np.array(timestamp)
        return record, visual, boxes

    def process(self, image, lidar, radar, timestamp, modality='all'):
        result = super().process(image, lidar, radar, timestamp, modality)
        result['metrics'].update(pipeline='middle', detector_schema='group8-middle-fusion-v1',
                                  detector_objects=len(result['lidar_proposals']),
                                  middle_camera_gate=self.middle_gate_mean[0],
                                  middle_lidar_gate=self.middle_gate_mean[1],
                                  middle_radar_gate=self.middle_gate_mean[2])
        for proposal in result['lidar_proposals']:
            proposal['origin'] = 'BEV fusion'
        result['detector_objects'] = result['lidar_proposals']
        return result
