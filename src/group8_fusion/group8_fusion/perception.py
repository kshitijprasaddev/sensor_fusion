"""Deterministic geometric detectors and object-level association."""
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from .geometry import box_corners, iou, project, transform_points


@dataclass
class Object3D:
    center: np.ndarray
    size: np.ndarray
    label: str = 'obstacle'
    confidence: float = 0.5
    sources: list = field(default_factory=lambda: ['lidar'])
    radial_velocity: float | None = None
    variance: np.ndarray = field(default_factory=lambda: np.array([0.16, 0.16]))


def dbscan(points, eps, min_points):
    """Density clusters using a KD tree. Noise and border points follow DBSCAN."""
    if len(points) == 0:
        return []
    tree = cKDTree(points)
    labels = np.full(len(points), -1, dtype=int)
    visited = np.zeros(len(points), bool)
    clusters = []
    for i in range(len(points)):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = tree.query_ball_point(points[i], eps)
        if len(neighbors) < min_points:
            continue
        label = len(clusters)
        queue = list(neighbors)
        queued = set(neighbors)
        labels[i] = label
        for j in queue:
            if not visited[j]:
                visited[j] = True
                near = tree.query_ball_point(points[j], eps)
                if len(near) >= min_points:
                    for k in near:
                        if k not in queued:
                            queue.append(k)
                            queued.add(k)
            if labels[j] == -1:
                labels[j] = label
        clusters.append(np.flatnonzero(labels == label))
    return clusters


def remove_ground(points, threshold=0.22):
    candidates = points[points[:, 2] < -0.65]
    if len(candidates) < 20:
        return points
    rng = np.random.default_rng(8)
    best = None
    score = 0
    for _ in range(70):
        a, b, c = candidates[rng.choice(len(candidates), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal /= norm
        if abs(normal[2]) < 0.92:
            continue
        offset = -normal @ a
        count = np.count_nonzero(np.abs(candidates @ normal + offset) < threshold)
        if count > score:
            best, score = (normal, offset), count
    if best is None:
        return points
    return points[np.abs(points @ best[0] + best[1]) > threshold]


def lidar_detect(raw, cfg):
    p = raw[np.isfinite(raw).all(axis=1)]
    lo, hi = np.asarray(cfg['roi_min']), np.asarray(cfg['roi_max'])
    p = p[((p > lo) & (p < hi)).all(axis=1) & (np.linalg.norm(p[:, :2], axis=1) > 2.0)]
    if len(p):
        _, indices = np.unique(np.floor(p / cfg['voxel']), axis=0, return_index=True)
        p = p[np.sort(indices)]
    p = remove_ground(p, cfg['ground_threshold'])
    objects = []
    for indices in dbscan(p, cfg['cluster_eps'], cfg['min_points']):
        cluster = p[indices]
        lo, hi = np.percentile(cluster, [2, 98], axis=0)
        size = hi - lo
        if size[2] < 0.35 or size[2] > 4 or max(size[:2]) > 10:
            continue
        objects.append(Object3D((lo + hi) / 2, np.maximum(size, 0.3)))
    return objects, p


def radar_detect(raw, lidar_from_radar, cfg):
    valid = np.isfinite(raw[:, :3]).all(axis=1)
    p = transform_points(raw[valid, :3], lidar_from_radar)
    velocity = raw[valid, 3]
    keep = (p[:, 0] > 2) & (p[:, 0] < cfg['max_range'])
    keep &= (np.abs(p[:, 1]) < cfg['max_lateral']) & (np.abs(p[:, 2]) < 3)
    p, velocity = p[keep], velocity[keep]
    objects = []
    for indices in dbscan(p[:, :2], cfg['cluster_eps'], cfg['min_points']):
        center = np.median(p[indices], axis=0)
        v = velocity[indices]
        v = v[np.isfinite(v)]
        objects.append(Object3D(center, np.array([1., 1., 1.]), 'radar_target', 0.35,
                                ['radar'], float(np.median(v)) if len(v) else None,
                                np.array([0.64, 2.25])))
    return objects, p


def associate_camera(objects, boxes, camera_from_lidar, K, width, height):
    if not objects or not boxes:
        return 0
    scores = np.zeros((len(objects), len(boxes)))
    for i, obj in enumerate(objects):
        corners = box_corners(obj.center, obj.size)
        pixels, _, depth = project(corners, camera_from_lidar, K, width, height)
        if np.all(depth > 0.1) and np.isfinite(pixels).all():
            box = np.r_[pixels.min(axis=0), pixels.max(axis=0)]
            for j, detection in enumerate(boxes):
                scores[i, j] = iou(box, np.asarray(detection['xyxy']))
    rows, cols = linear_sum_assignment(-scores)
    count = 0
    for i, j in zip(rows, cols):
        if scores[i, j] >= 0.05:
            obj, det = objects[i], boxes[j]
            obj.label, obj.confidence = det['label'], det['confidence']
            obj.sources.append('camera')
            count += 1
    return count


def associate_radar(objects, radar, gate=2.5):
    used = set()
    if objects and radar:
        costs = np.array([[np.linalg.norm(a.center[:2] - b.center[:2]) for b in radar]
                          for a in objects])
        # Make forbidden edges more expensive than any combination of valid ones.
        rows, cols = linear_sum_assignment(np.where(costs <= gate, costs, 1e6))
        for i, j in zip(rows, cols):
            if costs[i, j] > gate:
                continue
            obj, measurement = objects[i], radar[j]
            variance = 1 / (1 / obj.variance + 1 / measurement.variance)
            obj.center[:2] = variance * (obj.center[:2] / obj.variance +
                                         measurement.center[:2] / measurement.variance)
            obj.variance = variance
            obj.radial_velocity = measurement.radial_velocity
            obj.sources.append('radar')
            used.add(j)
    objects.extend(obj for j, obj in enumerate(radar) if j not in used)
    return len(used)
