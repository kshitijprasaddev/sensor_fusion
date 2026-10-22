"""Neural perception profile: pretrained encoders, learned fusion and SAM identities."""

from pathlib import Path
import hashlib
import time
from types import SimpleNamespace
import numpy as np
import torch
from .attention_fusion import pack_features, CAMERA_CLASSES
from .robust_fusion import load_fusion_model
from .neural_models import VoxelNeXtDetector, SAMMemoryTracker
from .perception import Object3D
from .processor import Processor


class NeuralProcessor:
    lidar_fields = ("x", "y", "z", "intensity")
    radar_fields = ("x", "y", "z", "v", "RCS")
    annotate = Processor.annotate

    def __init__(
        self,
        cfg,
        lidar_from_camera,
        lidar_from_radar,
        detector=None,
        training_features=False,
    ):
        self.cfg = cfg
        self.camera_from_lidar = np.linalg.inv(lidar_from_camera)
        self.lidar_from_radar = lidar_from_radar
        self.K = np.asarray(cfg["camera"]["K"]).reshape(3, 3)
        self.detector = detector
        checkpoint = Path(cfg["neural"]["fusion"]["weights"])
        if not training_features and not checkpoint.is_file():
            raise FileNotFoundError(
                "Train the learned fusion head first: ./scripts/run.sh ai-train"
            )
        self.lidar_detector = VoxelNeXtDetector(cfg["neural"]["lidar"])
        self.visual_tracker = SAMMemoryTracker(cfg["neural"]["tracking"])
        self.last_time = None
        self.tracks = []
        self.hidden = {}
        self.previous = {}
        self.tracker = self
        self.network = None
        if not training_features:
            checkpoint = Path(cfg["neural"]["fusion"]["weights"])
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    "Train the learned fusion head first: ./scripts/run.sh ai-train"
                )
            data = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if len(data.get("camera_intrinsics", [])) != 9 or not np.allclose(
                data["camera_intrinsics"], cfg["camera"]["K"]
            ):
                raise ValueError(
                    "Camera intrinsics changed: re-extract features and retrain the fusion head"
                )
            self.network = load_fusion_model(data)
            self.fusion_schema = data["schema"]
            self.network.cuda().eval()
            self.checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def reset_tracking(self):
        self.visual_tracker.reset()
        self.last_time = None
        self.tracks = []
        self.hidden.clear()
        self.previous.clear()

    def extract(self, image, lidar, radar, timestamp):
        if self.last_time is not None and (
            timestamp <= self.last_time or timestamp - self.last_time > 2
        ):
            self.reset_tracking()
        self.last_time = timestamp
        if image.shape[1::-1] != (
            self.cfg["camera"]["width"],
            self.cfg["camera"]["height"],
        ):
            raise ValueError("Image dimensions differ from camera calibration")
        boxes = self.detector.detect(image)
        visual = self.visual_tracker.update(image, boxes)
        detections = self.lidar_detector.detect(lidar)
        raw = np.empty((0, 5), np.float32) if radar is None else radar
        record = pack_features(
            visual,
            detections,
            raw,
            self.lidar_from_radar,
            self.camera_from_lidar,
            self.K,
            image.shape[1],
            image.shape[0],
        )
        record["time"] = np.array(timestamp)
        return record, visual, boxes

    def infer_record(self, record):
        if self.network is None:
            raise RuntimeError("Untrained fusion is disabled")
        if not len(record["query"]):
            return [], {}
        tensor = lambda key: torch.as_tensor(
            record[key], dtype=torch.float32, device="cuda"
        )
        with torch.inference_mode():
            fused, la, ra = self.network(
                tensor("query"), tensor("lidar"), tensor("radar"), tensor("pair")
            )
            lp = la.cpu().numpy()
            rp = ra.cpu().numpy()
            selected = lp.argmax(1)
            # Confidence and null-token rejection prevent forced associations.
            positions = []
            previous = []
            times = []
            hidden = []
            accepted = []
            timestamp = float(record["time"])
            for j, index in enumerate(selected):
                if (
                    index == len(record["boxes"])
                    or lp[j, index] < self.cfg["neural"]["fusion"]["min_confidence"]
                ):
                    continue
                identifier = int(record["ids"][j])
                pos = record["boxes"][index, :3]
                old = self.previous.get(identifier)
                dt = timestamp - old[0] if old else 0
                if dt < 0 or dt > 2:
                    old = None
                    dt = 0
                positions.append(pos)
                # Match the zero/no-history sentinel used during head training.
                previous.append(old[1] if old else np.zeros(3, np.float32))
                times.append(dt)
                hidden.append(
                    self.hidden.get(identifier, torch.zeros(self.network.dimension, device="cuda"))
                    if old else torch.zeros(self.network.dimension, device="cuda")
                )
                accepted.append(j)
            if not accepted:
                return [], {"neural_null_tracks": len(record["ids"])}
            position = torch.as_tensor(np.asarray(positions), device="cuda")
            velocity, state = self.network.predict_motion(
                fused[accepted],
                position,
                torch.as_tensor(np.asarray(previous), device="cuda"),
                torch.as_tensor(times, device="cuda"),
                torch.stack(hidden),
            )
            velocity = velocity.cpu().numpy()
        tracks = []
        radar_matches = 0
        # Independent learned selections can collide. Keep the highest-posterior
        # query for each LiDAR box; this is suppression, not a global assignment.
        best = {}
        for j in accepted:
            index = int(selected[j])
            if index not in best or lp[j, index] > lp[best[index], index]:
                best[index] = j
        for row, j in enumerate(accepted):
            index = int(selected[j])
            if best[index] != j:
                continue
            identifier = int(record["ids"][j])
            box = record["boxes"][index]
            obj = Object3D(
                box[:3].astype(float).copy(),
                box[3:6].astype(float).copy(),
                CAMERA_CLASSES[int(record["labels"][j])],
                float(lp[j, index]),
                ["camera", "lidar"],
            )
            obj.yaw = float(box[6])
            rindex = int(rp[j].argmax())
            obj.radial_velocity = None
            if (
                rindex < len(record["radar_xyz"])
                and rp[j, rindex] >= self.cfg["neural"]["fusion"].get(
                    "min_radar_confidence", self.cfg["neural"]["fusion"]["min_confidence"]
                )
            ):
                # Conservative range sanity gate; geometric safety is not itself
                # a neural algorithm and is disclosed separately from attention.
                if np.linalg.norm(record["radar_xyz"][rindex, :2] - box[:2]) < 5:
                    obj.sources.append("radar")
                    radar_matches += 1
                    rv = float(record["radar_v"][rindex])
                    obj.radial_velocity = rv if np.isfinite(rv) else None
            v = velocity[row] if times[row] > 0 else np.zeros(2)
            self.previous[identifier] = (timestamp, box[:3].copy())
            self.hidden[identifier] = state[row].detach()
            tracks.append(
                SimpleNamespace(
                    id=identifier,
                    state=np.r_[obj.center[:2], v],
                    # Fixed reporting covariance is heuristic, not calibrated uncertainty.
                    covariance=np.diag([1.0, 1.0, 9.0, 9.0]),
                    object=obj,
                    measured=True,
                )
            )
        self.tracks = tracks
        return tracks, {
            "neural_null_tracks": len(record["ids"]) - len(tracks),
            "radar_matches": radar_matches,
            "mean_lidar_attention": float(lp.max(1).mean()),
            "mean_radar_attention": float(rp.max(1).mean()),
        }

    def process(self, image, lidar, radar, timestamp, modality="all"):
        if modality != "all":
            raise ValueError(
                "Neural profile currently supports all; classical ablation modes are separate"
            )
        started = time.perf_counter()
        record, visual, boxes = self.extract(image, lidar, radar, timestamp)
        tracks, extra = self.infer_record(record)
        points = lidar[:, :3]
        points = points[np.isfinite(points).all(1)]
        # Visualization sampling only; no RANSAC or clustering in neural profile.
        points = points[
            (np.abs(points[:, 1]) < 25) & (points[:, 0] > -5) & (points[:, 0] < 65)
        ][::3]
        radar_matches = extra.get("radar_matches", 0)
        metrics = {
            "time": timestamp,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "camera_detections": len(boxes),
            "lidar_objects": len(record["boxes"]),
            "radar_objects": len(record["radar_xyz"]),
            "camera_matches": len(tracks),
            "radar_matches": radar_matches,
            "tracks": len(tracks),
            "measured_tracks": len(tracks),
            "three_sensor_objects": radar_matches,
            "pipeline": "neural",
            "fusion_schema": self.fusion_schema,
            "radar_count_unit": "filtered_points_not_clusters",
            "neural_null_tracks": len(visual) - len(tracks),
            "mean_lidar_attention": 0.0,
            "mean_radar_attention": 0.0,
            "visual_tracks": len(visual),
            "sam_capacity_rejections": self.visual_tracker.capacity_rejections,
            **extra,
        }
        # Camera masks/IDs remain visible even when no valid 3D observation exists.
        return {
            "objects": [t.object for t in tracks],
            "tracks": tracks,
            "boxes": boxes,
            "nonground": points,
            "radar_points": record["radar_xyz"],
            "metrics": metrics,
            "visual_tracks": visual,
        }
