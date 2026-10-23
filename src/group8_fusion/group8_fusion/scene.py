"""RViz geometry for measured tracks, sensor-relative guides and proposals."""

import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from .geometry import BOX_EDGES, oriented_corners
from .display import track_rgb, source_label


def marker(header, namespace, identifier, kind, rgb, alpha=1.0):
    out = Marker(
        header=header, ns=namespace, id=int(identifier), type=kind, action=Marker.ADD
    )
    out.pose.orientation.w = 1.0
    out.color.r, out.color.g, out.color.b = map(float, rgb)
    out.color.a = float(alpha)
    # Simulation-time expiry; explicit DELETEALL below also handles disappearance
    # and rewinds without relying on timestamps increasing monotonically.
    out.lifetime.nanosec = 400000000
    return out


def line(
    header, namespace, identifier, points, rgb, width=0.045, strip=False, alpha=1.0
):
    out = marker(
        header,
        namespace,
        identifier,
        Marker.LINE_STRIP if strip else Marker.LINE_LIST,
        rgb,
        alpha,
    )
    out.scale.x = float(width)
    out.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in points]
    return out


def text(header, namespace, identifier, position, label, rgb, size=0.36):
    out = marker(header, namespace, identifier, Marker.TEXT_VIEW_FACING, rgb)
    out.pose.position.x, out.pose.position.y, out.pose.position.z = map(float, position)
    out.scale.z = float(size)
    out.text = label
    return out


class SceneRenderer:
    def __init__(self, configuration=None):
        self.cfg = configuration or {}

    def guides(self, header):
        z = float(self.cfg.get("reference_z", -1.7))
        result = []
        angles = np.linspace(0, 2 * np.pi, 129)
        for radius in (10, 20, 40, 60):
            points = np.c_[
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.full(len(angles), z),
            ]
            result.append(
                line(
                    header,
                    "distance_rings",
                    radius,
                    points,
                    (0.24, 0.34, 0.43),
                    0.025,
                    True,
                    0.55,
                )
            )
            result.append(
                text(
                    header,
                    "distance_labels",
                    radius,
                    (radius, -1.0, z + 0.1),
                    f"{radius} m",
                    (0.60, 0.72, 0.82),
                    0.32,
                )
            )
        # A reference symbol, not a surveyed vehicle outline or ground estimate.
        result.append(
            line(
                header,
                "rig_origin",
                0,
                [(-0.3, 0, z), (0.3, 0, z), (0, 0, z), (0, -0.3, z), (0, 0.3, z)],
                (0.7, 0.8, 0.9),
                0.04,
                True,
            )
        )
        result.append(
            text(
                header,
                "rig_origin",
                1,
                (0, -1.3, z + 0.15),
                "RIG",
                (0.7, 0.8, 0.9),
                0.22,
            )
        )
        return result

    def tracks(self, header, tracks, layer="labels"):
        if layer not in ("labels", "boxes", "motion"):
            raise ValueError("Unknown scene layer")
        result = [Marker(action=Marker.DELETEALL)]
        if layer == "labels" and self.cfg.get("range_rings", True):
            result.extend(self.guides(header))
        for track in tracks:
            obj = track.object
            if not self.cfg.get("show_unclassified", False) and obj.label in (
                "obstacle",
                "radar_target",
            ):
                continue
            center = np.r_[track.state[:2], obj.center[2]].astype(float)
            size = np.asarray(obj.size, float)
            yaw = float(getattr(obj, "yaw", 0.0))
            if (
                not np.isfinite(np.r_[center, size, yaw, track.state[2:]]).all()
                or (size <= 0).any()
            ):
                continue
            rgb = track_rgb(track.id)
            corners = oriented_corners(center, size, yaw)
            edges = [corners[i] for edge in BOX_EDGES for i in edge]
            alpha = 1.0 if track.measured else 0.4
            result.append(
                line(header, "box_outlines", track.id, edges, rgb, 0.055, alpha=alpha)
            )
            cube = marker(
                header,
                "box_fill",
                track.id,
                Marker.CUBE,
                rgb,
                0.055 if track.measured else 0.02,
            )
            cube.pose.position.x, cube.pose.position.y, cube.pose.position.z = map(
                float, center
            )
            cube.pose.orientation.z, cube.pose.orientation.w = float(
                np.sin(yaw / 2)
            ), float(np.cos(yaw / 2))
            cube.scale.x, cube.scale.y, cube.scale.z = map(float, size)
            result.append(cube)
            # Bottom face helps read box dimensions in the bird's-eye view.
            result.append(
                line(
                    header,
                    "box_footprints",
                    track.id,
                    corners[[0, 4, 6, 2, 0]],
                    rgb,
                    0.075,
                    True,
                    alpha,
                )
            )
            distance = float(np.linalg.norm(center[:2]))
            label = f"#{track.id:02d} {obj.label.upper()} | {distance:.1f} m"
            if self.cfg.get("detailed_labels", False):
                label += '\n' + source_label(obj.sources)
            if not track.measured:
                label += " | predicted"
            # Move labels towards the rig centerline, not beyond the outside of
            # a near-side object. Close objects also receive smaller labels.
            side = -1. if center[1] > 0 else 1.
            anchor = center + [0, side * (size[1] / 2 + 1.5), size[2] / 2 + 0.8]
            label_size = self.cfg.get("label_size", .32) * np.clip(distance / 12., .6, 1.)
            result.append(
                text(
                    header,
                    "track_labels",
                    track.id,
                    anchor,
                    label,
                    rgb,
                    label_size,
                )
            )
            result.append(
                line(
                    header,
                    "label_leaders",
                    track.id,
                    [center + [0, 0, size[2] / 2], anchor - [0, 0, 0.4]],
                    rgb,
                    0.018,
                    True,
                    0.6,
                )
            )
            # Detector yaw is distinct from direction of motion.
            forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            side = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
            tip = center + forward * size[0] / 2
            tip[2] = center[2] - size[2] / 2 + 0.08
            result.append(
                line(
                    header,
                    "box_heading",
                    track.id,
                    [
                        tip - 0.5 * forward + 0.3 * side,
                        tip,
                        tip - 0.5 * forward - 0.3 * side,
                    ],
                    rgb,
                    0.075,
                    True,
                )
            )
            speed = float(np.linalg.norm(track.state[2:]))
            if speed >= self.cfg.get("velocity_min_speed", 1.0):
                velocity = np.r_[track.state[2:], 0.0].astype(float)
                arrow = marker(header, "relative_velocity", track.id, Marker.ARROW, rgb)
                horizon = self.cfg.get("velocity_horizon_s", 0.75)
                length = self.cfg.get("velocity_max_length", 4.0)
                start = center.copy()
                start[2] = float(self.cfg.get("reference_z", -1.7)) + 0.15
                end = start + velocity * min(horizon, length / speed)
                arrow.points = [
                    Point(x=float(x), y=float(y), z=float(z))
                    for x, y, z in (start, end)
                ]
                arrow.scale.x, arrow.scale.y, arrow.scale.z = (
                    0.045,
                    0.12,
                    min(0.3, speed * horizon * 0.3),
                )
                result.append(arrow)
                result.append(
                    text(
                        header,
                        "relative_velocity_labels",
                        track.id,
                        end + [0, 0, 0.35],
                        f"v_rel estimate {speed:.1f} m/s",
                        rgb,
                        0.3,
                    )
                )
        boxes = {"box_outlines", "box_fill", "box_footprints", "box_heading"}
        motion = {"relative_velocity", "relative_velocity_labels"}
        if layer == "boxes":
            result = [
                m for m in result if m.action == Marker.DELETEALL or m.ns in boxes
            ]
        elif layer == "motion":
            result = [
                m for m in result if m.action == Marker.DELETEALL or m.ns in motion
            ]
        else:
            result = [
                m
                for m in result
                if m.action == Marker.DELETEALL or m.ns not in boxes | motion
            ]
        return MarkerArray(markers=result)

    def proposals(self, header, proposals):
        result = [Marker(action=Marker.DELETEALL)]
        for index, proposal in enumerate(proposals):
            box = np.asarray(proposal["box"])
            if len(box) < 7 or not np.isfinite(box[:7]).all() or (box[3:6] <= 0).any():
                continue
            corners = oriented_corners(box[:3], box[3:6], box[6])
            result.append(
                line(
                    header,
                    ("middle_fusion_boxes" if proposal.get("origin") == "BEV fusion" else "unfused_lidar_boxes"),
                    index,
                    [corners[i] for edge in BOX_EDGES for i in edge],
                    (0.57, 0.65, 0.75),
                    0.025,
                    alpha=0.55,
                )
            )
            result.append(
                text(
                    header,
                    ("middle_fusion_labels" if proposal.get("origin") == "BEV fusion" else "unfused_lidar_labels"),
                    index,
                    box[:3] + [0, 0, box[5] / 2 + 0.3],
                    f"{proposal.get('origin', 'LiDAR only')}: {proposal['label']}",
                    (0.65, 0.73, 0.8),
                    0.25,
                )
            )
        return MarkerArray(markers=result)
