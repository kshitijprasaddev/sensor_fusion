"""Shared identity colors and bounded camera drawing. No perception changes."""

import cv2
import numpy as np
from .geometry import BOX_EDGES, oriented_corners, project


PALETTE = (
    (53, 222, 201),
    (255, 190, 70),
    (166, 139, 255),
    (255, 125, 163),
    (104, 198, 255),
    (218, 227, 95),
    (239, 159, 85),
    (191, 217, 243),
)


def track_rgb(identifier):
    return tuple(
        channel / 255 for channel in PALETTE[(int(identifier) - 1) % len(PALETTE)]
    )


def track_bgr(identifier):
    return PALETTE[(int(identifier) - 1) % len(PALETTE)][::-1]


def source_label(sources):
    return "+".join(
        name
        for key, name in (("camera", "CAM"), ("lidar", "LID"), ("radar", "RAD"))
        if key in sources
    )


def text_tag(image, text, anchor, color, scale=0.75):
    """Keep the complete label inside the image, including near image borders."""
    height, width = image.shape[:2]
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = int(np.clip(anchor[0], 4, max(4, width - tw - 12)))
    y = int(np.clip(anchor[1], th + 12, max(th + 12, height - baseline - 8)))
    cv2.rectangle(
        image,
        (x - 4, y - th - 6),
        (min(width - 1, x + tw + 6), y + baseline + 5),
        (18, 25, 34),
        -1,
    )
    cv2.putText(
        image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA
    )


def draw_projected_box(image, center, size, yaw, transform, K, color):
    corners = oriented_corners(center, size, yaw)
    uv, _, depth = project(corners, transform, K, image.shape[1], image.shape[0])
    # Skip near-plane intersections rather than drawing enormous crossing edges.
    if not (depth > 0.1).all() or not np.isfinite(uv).all():
        return
    pixels = np.clip(np.rint(uv), -100000, 100000).astype(int)
    for a, b in BOX_EDGES:
        visible, start, end = cv2.clipLine(
            (0, 0, image.shape[1], image.shape[0]), tuple(pixels[a]), tuple(pixels[b])
        )
        if visible:
            cv2.line(image, start, end, color, 3, cv2.LINE_AA)


def annotate_frame(processor, image, result):
    out = image.copy()
    cfg = processor.cfg.get("visualization", {})
    tracks = {track.id: track for track in result["tracks"]}
    visual = result.get("visual_tracks", [])
    if cfg.get("camera_points", False):
        # Optional diagnostic, bounded so the scene is not buried in dots.
        points = result["nonground"]
        points = points[:: max(1, int(np.ceil(len(points) / 2000)))]
        pixels, valid, _ = project(
            points,
            processor.camera_from_lidar,
            processor.K,
            image.shape[1],
            image.shape[0],
        )
        for u, v in pixels[valid]:
            cv2.circle(out, (int(u), int(v)), 1, (170, 150, 90), -1)
    for instance in visual:
        color = track_bgr(instance["id"])
        mask = instance["mask"]
        out[mask] = (out[mask].astype(float) * 0.8 + np.asarray(color) * 0.2).astype(
            np.uint8
        )
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, color, 3, cv2.LINE_AA)
    if cfg.get("projected_boxes", False):
        for track in tracks.values():
            if not cfg.get("show_unclassified", False) and track.object.label in (
                "obstacle",
                "radar_target",
            ):
                continue
            draw_projected_box(
                out,
                np.r_[track.state[:2], track.object.center[2]],
                track.object.size,
                getattr(track.object, "yaw", 0.0),
                processor.camera_from_lidar,
                processor.K,
                track_bgr(track.id),
            )
    # With SAM identities present, YOLO proposals are optional to avoid duplicate
    # boxes and contradictory color meanings. They never disappear from metrics.
    if not visual or cfg.get("camera_proposals", False):
        for box in result["boxes"]:
            x1, y1, x2, y2 = map(int, box["xyxy"])
            cv2.rectangle(out, (x1, y1), (x2, y2), (70, 210, 255), 2)
            text_tag(
                out, f"YOLO {box['label']}", (x1, max(165, y1 - 12)), (70, 210, 255)
            )
    labelled = set()
    for instance in visual:
        track = tracks.get(instance["id"])
        label = f"#{instance['id']:02d} {instance['label'].upper()}"
        label += (
            f" | {np.linalg.norm(track.state[:2]):.1f} m | {source_label(track.object.sources)}"
            if track
            else " | visual only"
        )
        x, y = instance["bbox"][:2]
        text_tag(out, label, (x, max(165, y - 12)), track_bgr(instance["id"]))
        labelled.add(instance["id"])
    for track in tracks.values():
        if track.id in labelled or (
            not cfg.get("show_unclassified", False)
            and track.object.label in ("obstacle", "radar_target")
        ):
            continue
        uv, valid, _ = project(
            np.r_[track.state[:2], track.object.center[2]][None],
            processor.camera_from_lidar,
            processor.K,
            image.shape[1],
            image.shape[0],
        )
        if valid[0]:
            text_tag(
                out,
                f"#{track.id:02d} {track.object.label.upper()} | {np.linalg.norm(track.state[:2]):.1f} m",
                (uv[0, 0], max(165, uv[0, 1])),
                track_bgr(track.id),
            )
    m = result["metrics"]
    approximate = processor.cfg["camera"]["status"] != "calibrated"
    cv2.rectangle(out, (0, 0), (image.shape[1], 134), (18, 25, 34), -1)
    title = (
        f"GROUP 8 | {m.get('pipeline', 'classical').upper()} | ID COLORS MATCH 3D VIEW"
    )
    cv2.putText(
        out,
        title,
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (235, 242, 248),
        2,
        cv2.LINE_AA,
    )
    status = "APPROXIMATE FUSION CALIBRATION" if approximate else "CALIBRATED CAMERA"
    cv2.putText(
        out,
        status + " | moving velodyne frame | receipt-time sync",
        (24, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.73,
        (90, 200, 255) if approximate else (120, 235, 180),
        2,
        cv2.LINE_AA,
    )
    counters = f"Tracks {m['tracks']} | Radar-supported {m['radar_matches']} | YOLO {m['camera_detections']} | LiDAR boxes {m['lidar_objects']}"
    counters += f" | Core {m['latency_ms']:.0f} ms"
    if m.get("rejected_3d_observations", 0):
        counters += f" | 3D withheld {m['rejected_3d_observations']}"
    cv2.putText(
        out,
        counters,
        (24, 114),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (210, 222, 234),
        2,
        cv2.LINE_AA,
    )
    return out
