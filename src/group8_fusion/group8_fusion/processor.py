"""Shared ROS and offline inference path."""
from pathlib import Path
import time
import cv2
import numpy as np
import yaml
from .geometry import project
from .perception import lidar_detect, radar_detect, associate_camera, associate_radar
from .tracking import Tracker


def load_config(path):
    with open(path, encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if config.get('pipeline', 'classical') not in ('classical', 'neural'):
        raise ValueError('Unknown perception pipeline')
    camera = config['camera']
    K = np.array(camera['K'], dtype=float).reshape(3, 3)
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError('Invalid camera intrinsic matrix')
    if camera['status'] != 'calibrated' and not camera['allow_approximate']:
        raise ValueError('Camera intrinsics are approximate; explicitly allow them or supply calibration')
    if config['synchronization']['tolerance_ms'] <= 0:
        raise ValueError('Synchronization tolerance must be positive')
    return config


class CameraDetector:
    def __init__(self, cfg):
        from ultralytics import YOLO
        self.cfg = cfg
        path = Path(cfg['weights'])
        if not path.is_file():
            raise FileNotFoundError(f'Model not found: {path}. Run scripts/prepare_model.py first.')
        self.model = YOLO(str(path), task='detect')
        self.model.predict(np.zeros((640, 640, 3), np.uint8), imgsz=cfg['imgsz'],
                           device=cfg['device'], rect=False, verbose=False)

    def detect(self, image):
        result = self.model.predict(image, imgsz=self.cfg['imgsz'], conf=self.cfg['confidence'],
                                    device=self.cfg['device'], rect=False, verbose=False)[0]
        boxes = []
        for b, c, p in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(),
                           result.boxes.conf.cpu().numpy()):
            boxes.append({'xyxy': b.tolist(), 'label': result.names[int(c)], 'confidence': float(p)})
        return boxes


class Processor:
    lidar_fields = ('x', 'y', 'z')
    radar_fields = ('x', 'y', 'z', 'v')

    def __init__(self, cfg, lidar_from_camera, lidar_from_radar, detector=None):
        self.cfg = cfg
        self.camera_from_lidar = np.linalg.inv(lidar_from_camera)
        self.lidar_from_radar = lidar_from_radar
        self.K = np.array(cfg['camera']['K']).reshape(3, 3)
        self.detector = detector
        self.tracker = Tracker(**cfg['tracking'])

    def reset_tracking(self):
        self.tracker.tracks.clear()
        self.tracker.last_time = None

    def process(self, image, lidar, radar, timestamp, modality='all'):
        start = time.perf_counter()
        if image.shape[1::-1] != (self.cfg['camera']['width'], self.cfg['camera']['height']):
            raise ValueError('Image dimensions differ from camera calibration')
        boxes = self.detector.detect(image) if self.detector and modality in ('all', 'camera_lidar') else []
        objects, nonground = lidar_detect(lidar, self.cfg['lidar'])
        camera_matches = associate_camera(objects, boxes, self.camera_from_lidar, self.K,
                                           image.shape[1], image.shape[0])
        radar_points = np.empty((0, 3))
        radar_matches = 0
        radar_count = 0
        if radar is not None and modality in ('all', 'lidar_radar'):
            radar_objects, radar_points = radar_detect(radar, self.lidar_from_radar, self.cfg['radar'])
            radar_count = len(radar_objects)
            radar_matches = associate_radar(objects, radar_objects, self.cfg['radar']['association_gate'])
        tracks = self.tracker.update(objects, timestamp)
        elapsed = time.perf_counter() - start
        metrics = {'time': timestamp, 'latency_ms': elapsed * 1000, 'camera_detections': len(boxes),
                   'lidar_objects': sum('lidar' in o.sources for o in objects),
                   'radar_objects': radar_count, 'camera_matches': camera_matches,
                   'radar_matches': radar_matches, 'tracks': len(tracks),
                   'measured_tracks': sum(t.measured for t in tracks),
                   'three_sensor_objects': sum(len(o.sources) == 3 for o in objects)}
        return {'objects': objects, 'tracks': tracks, 'boxes': boxes, 'nonground': nonground,
                'radar_points': radar_points, 'metrics': metrics}

    def annotate(self, image, result):
        from .display import annotate_frame
        return annotate_frame(self, image, result)


def make_processor(cfg, lidar_from_camera, lidar_from_radar, detector=None):
    if cfg.get('pipeline', 'classical') == 'neural':
        from .neural import NeuralProcessor
        return NeuralProcessor(cfg, lidar_from_camera, lidar_from_radar, detector)
    return Processor(cfg, lidar_from_camera, lidar_from_radar, detector)
