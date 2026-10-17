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
                           device=cfg['device'], verbose=False)

    def detect(self, image):
        result = self.model.predict(image, imgsz=self.cfg['imgsz'], conf=self.cfg['confidence'],
                                    device=self.cfg['device'], verbose=False)[0]
        boxes = []
        for b, c, p in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(),
                           result.boxes.conf.cpu().numpy()):
            boxes.append({'xyxy': b.tolist(), 'label': result.names[int(c)], 'confidence': float(p)})
        return boxes


class Processor:
    def __init__(self, cfg, lidar_from_camera, lidar_from_radar, detector=None):
        self.cfg = cfg
        self.camera_from_lidar = np.linalg.inv(lidar_from_camera)
        self.lidar_from_radar = lidar_from_radar
        self.K = np.array(cfg['camera']['K']).reshape(3, 3)
        self.detector = detector
        self.tracker = Tracker(**cfg['tracking'])

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
        out = image.copy()
        pixels, valid, depths = project(result['nonground'], self.camera_from_lidar, self.K,
                                         image.shape[1], image.shape[0])
        for (u, v), depth in zip(pixels[valid][::2], depths[valid][::2]):
            color = (int(255 * min(depth / 60, 1)), 220, int(255 * (1 - min(depth / 60, 1))))
            cv2.circle(out, (int(u), int(v)), 2, color, -1)
        for box in result['boxes']:
            x1, y1, x2, y2 = map(int, box['xyxy'])
            cv2.rectangle(out, (x1, y1), (x2, y2), (80, 220, 255), 2)
            cv2.putText(out, f"{box['label']} {box['confidence']:.2f}", (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 255), 2)
        for track in result['tracks']:
            xyz = np.r_[track.state[:2], track.object.center[2]].reshape(1, 3)
            pixel, valid, _ = project(xyz, self.camera_from_lidar, self.K, image.shape[1], image.shape[0])
            if valid[0]:
                u, v = pixel[0].astype(int)
                label = f'#{track.id} {track.object.label} {np.linalg.norm(track.state[:2]):.1f}m'
                cv2.putText(out, label, (u, v), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (90, 255, 100), 2)
        cv2.rectangle(out, (0, 0), (image.shape[1], 100), (22, 28, 38), -1)
        status = 'APPROXIMATE CAMERA INTRINSICS' if self.cfg['camera']['status'] != 'calibrated' else 'CALIBRATED'
        cv2.putText(out, f'GROUP 8 | Camera + LiDAR + Radar | {status}', (24, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2)
        m = result['metrics']
        cv2.putText(out, f"YOLO: {m['camera_detections']} | Camera matches: {m['camera_matches']} | Radar matches: {m['radar_matches']} | Tracks: {m['tracks']}",
                    (24, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 220, 255), 2)
        return out
