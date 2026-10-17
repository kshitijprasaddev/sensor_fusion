"""Constant-velocity Kalman tracking in the vehicle-mounted LiDAR frame."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    id: int
    state: np.ndarray
    covariance: np.ndarray
    object: object
    last_seen: float
    hits: int = 1
    measured: bool = True


class Tracker:
    def __init__(self, gate=3.0, max_age=0.7, min_hits=2):
        self.gate, self.max_age, self.min_hits = gate, max_age, min_hits
        self.tracks = []
        self.next_id = 1
        self.last_time = None

    def update(self, objects, timestamp):
        if self.last_time is not None and (timestamp <= self.last_time or timestamp - self.last_time > 2):
            self.tracks.clear()
        dt = 0 if self.last_time is None else max(0, timestamp - self.last_time)
        self.last_time = timestamp
        self.tracks = [t for t in self.tracks if timestamp - t.last_seen <= self.max_age]
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        Q = G @ G.T * 4 + np.eye(4) * 1e-5
        H = np.eye(2, 4)
        for track in self.tracks:
            track.state = F @ track.state
            track.covariance = F @ track.covariance @ F.T + Q
            track.measured = False
        used = set()
        if self.tracks and objects:
            costs = np.full((len(self.tracks), len(objects)), 1e6)
            for i, track in enumerate(self.tracks):
                for j, obj in enumerate(objects):
                    innovation = obj.center[:2] - H @ track.state
                    S = H @ track.covariance @ H.T + np.diag(obj.variance)
                    mahalanobis = innovation @ np.linalg.solve(S, innovation)
                    if np.linalg.norm(innovation) <= self.gate and mahalanobis <= 9.21:
                        costs[i, j] = mahalanobis
            rows, cols = linear_sum_assignment(costs)
            for i, j in zip(rows, cols):
                if costs[i, j] >= 1e6:
                    continue
                track, obj = self.tracks[i], objects[j]
                R = np.diag(obj.variance)
                S = H @ track.covariance @ H.T + R
                gain = np.linalg.solve(S, H @ track.covariance).T
                track.state += gain @ (obj.center[:2] - H @ track.state)
                A = np.eye(4) - gain @ H
                track.covariance = A @ track.covariance @ A.T + gain @ R @ gain.T
                # Keep a semantic label through temporary camera misses.
                if obj.label in ('obstacle', 'radar_target') and track.object.label not in ('obstacle', 'radar_target'):
                    obj.label = track.object.label
                track.object, track.last_seen = obj, timestamp
                track.hits += 1
                track.measured = True
                used.add(j)
        for j, obj in enumerate(objects):
            if j not in used:
                self.tracks.append(Track(self.next_id, np.r_[obj.center[:2], 0., 0.],
                                         np.diag([*obj.variance, 16., 16.]), obj, timestamp))
                self.next_id += 1
        return [t for t in self.tracks if t.hits >= self.min_hits]
