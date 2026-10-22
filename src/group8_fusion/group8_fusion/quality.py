"""Explicit runtime validity checks, independent of neural confidence."""

import numpy as np


def observation_issue(box, timestamp, previous, configuration):
    box = np.asarray(box)
    if (
        not np.isfinite(timestamp)
        or len(box) < 7
        or not np.isfinite(box[:7]).all()
        or (box[3:6] <= 0).any()
    ):
        return "invalid_geometry"
    if previous is not None and configuration.get("enabled", True):
        dt = timestamp - previous[0]
        if dt <= 0:
            return "nonmonotonic_time"
        if dt <= configuration.get("continuity_window_s", 0.5):
            allowed = (
                configuration.get("position_slack_m", 1.0)
                + configuration.get("max_relative_speed_mps", 30.0) * dt
            )
            if np.linalg.norm(box[:2] - previous[1][:2]) > allowed:
                return "position_jump"
    return None


def instance_points(points, tracks):
    """Color real points inside matched oriented boxes, not 3D segmentation.

    In overlapping boxes, the nearest normalized box center owns a point. No
    synthetic surfaces, interpolation or moving-frame history is added.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    points = points[np.isfinite(points).all(1)]
    owner = np.full(len(points), -1, dtype=np.int64)
    cost = np.full(len(points), np.inf)
    for track in tracks:
        obj = track.object
        center = np.r_[track.state[:2], obj.center[2]]
        size = np.asarray(obj.size)
        yaw = float(getattr(obj, "yaw", 0.0))
        if not np.isfinite(np.r_[center, size, yaw]).all() or (size <= 0).any():
            continue
        c, s = np.cos(yaw), np.sin(yaw)
        local = (points - center) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        normalized = local / (size / 2)
        distance = (normalized**2).sum(1)
        take = (np.abs(normalized) <= 1.0).all(1) & (distance < cost)
        cost[take] = distance[take]
        owner[take] = int(track.id)
    valid = owner >= 0
    return points[valid], owner[valid]
