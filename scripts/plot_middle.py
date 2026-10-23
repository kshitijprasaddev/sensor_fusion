#!/usr/bin/env python3
"""Plot real cached BEV inputs, learned gates and independent middle predictions.

This diagnostic runs on CPU. It loads the saved head's state dict directly;
it does not invoke sensor encoders, ROS, a GPU or teacher detection at runtime.
Teacher boxes are shown only as the recorded pseudo-label reference.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "group8_fusion"))
from group8_fusion.mid_model import BEVMiddleFusion, decode
from group8_fusion.mid_training import teacher_targets

NAMES = (
    "car", "truck", "construction vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic cone",
)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def footprint(box):
    """Return the actual oriented box's XY corners, without display smoothing."""
    corners = np.array([[1., 1.], [1., -1.], [-1., -1.], [-1., 1.]])
    corners *= box[None, 3:5] / 2
    cosine, sine = np.cos(box[6]), np.sin(box[6])
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return corners @ rotation.T + box[None, :2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path,
                        default=ROOT / "results/middle_features/studentProject/00000.npz")
    parser.add_argument("--weights", type=Path, default=ROOT / "models/middle_fusion.pt")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/assets/middle-fusion-features.png")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the saved detection threshold for this figure only")
    parser.add_argument("--teacher-threshold", type=float, default=.35)
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    grid = checkpoint["grid"]
    architecture = checkpoint.get("architecture", {})
    model = BEVMiddleFusion(**architecture).cpu().eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    threshold = (checkpoint["score_threshold"] if args.threshold is None
                 else args.threshold)
    if not 0 <= threshold <= 1 or not 0 <= args.teacher_threshold <= 1:
        parser.error("Both score thresholds must be probabilities in [0, 1]")

    channels = dict(camera=192, lidar=128, radar=8, availability=3)
    with np.load(args.frame, allow_pickle=False) as cached:
        arrays = {key: cached[key].astype(np.float32) for key in channels}
        frame = int(cached["frame"])
        time = float(cached["time"])
        teachers, teacher_labels, _ = teacher_targets(
            cached["teacher_boxes"], cached["teacher_labels"],
            cached["teacher_scores"], grid, threshold=args.teacher_threshold,
        )
    for key, count in channels.items():
        expected = (count, grid["height"], grid["width"])
        if arrays[key].shape != expected or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Expected finite {key} features with shape {expected}")
    inputs = {key: torch.from_numpy(value)[None] for key, value in arrays.items()}
    with torch.inference_mode():
        prediction = model(**inputs)
        if not all(torch.isfinite(value).all() for value in prediction.values()):
            raise FloatingPointError("The trained head produced non-finite values")
        decoded = decode(prediction, grid, threshold=threshold)
        gates = prediction["gates"][0].permute(1, 2, 0).numpy()
        heatmap = prediction["heatmap_logits"][0].sigmoid().amax(0).numpy()

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.labelcolor": "#475569", "xtick.color": "#475569",
        "ytick.color": "#475569", "axes.edgecolor": "#cbd5e1",
        "figure.facecolor": "white", "axes.facecolor": "#f8fafc",
    })
    figure, axes = plt.subplots(2, 3, figsize=(16, 10.6), layout="constrained")
    extent = (grid["x_min"], grid["x_min"] + grid["width"] * grid["cell_size"],
              grid["y_min"], grid["y_min"] + grid["height"] * grid["cell_size"])
    for axis in axes.flat:
        axis.set(xlim=extent[:2], ylim=extent[2:],
                 xlabel="Forward x (m)", ylabel="Lateral y (m)")
        axis.set_aspect("equal")
        axis.set_xticks(np.linspace(extent[0], extent[1], 5))
        axis.set_yticks(np.linspace(extent[2], extent[3], 5))
        axis.tick_params(length=3, labelsize=8)

    maps = (
        (axes[0, 0], np.linalg.norm(arrays["camera"], axis=0),
         "Camera spatial feature magnitude", "L2 norm across 192 channels", "viridis"),
        (axes[0, 1], np.linalg.norm(arrays["lidar"], axis=0),
         "LiDAR spatial feature magnitude", "L2 norm across 128 channels", "viridis"),
        (axes[0, 2], arrays["radar"][0], "Radar measured-return density",
         "Raster channel 0: clipped log density", "magma"),
    )
    for axis, values, title, color_label, cmap in maps:
        scale = {"vmin": 0}
        if axis is axes[0, 2]:
            # Use the measured range so sparse single returns remain visible;
            # the colorbar retains the actual normalized channel values.
            scale["vmax"] = max(float(values.max()), .01)
        image = axis.imshow(values, extent=extent, origin="lower", cmap=cmap,
                            interpolation="nearest", **scale)
        axis.set_title(title)
        colorbar = figure.colorbar(image, ax=axis, fraction=.046, pad=.025)
        colorbar.set_label(color_label, fontsize=8)
        colorbar.ax.tick_params(labelsize=7)

    axes[1, 0].imshow(np.clip(gates, 0, 1), extent=extent, origin="lower",
                      interpolation="nearest")
    axes[1, 0].set_title("Learned mixing weights at each cell")
    axes[1, 0].legend(
        handles=[Patch(color=color, label=label) for color, label in
                 [("#ef4444", "Camera (R)"), ("#22c55e", "LiDAR (G)"),
                  ("#3b82f6", "Radar (B)")]],
        loc="upper right", fontsize=7, framealpha=.95,
    )
    image = axes[1, 1].imshow(heatmap, extent=extent, origin="lower", cmap="inferno",
                              vmin=0, vmax=1, interpolation="nearest")
    axes[1, 1].set_title("New detector: maximum class-center score")
    colorbar = figure.colorbar(image, ax=axes[1, 1], fraction=.046, pad=.025)
    colorbar.set_label("Sigmoid output; not calibrated accuracy", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    axis = axes[1, 2]
    axis.set_title(f"Own decoded boxes and teacher reference\nSaved threshold {threshold:g}")
    axis.grid(color="#cbd5e1", alpha=.5, linewidth=.6)
    teacher_color, prediction_color = "#2563eb", "#c2410c"
    for box in teachers:
        axis.add_patch(Polygon(footprint(box), closed=True, fill=False,
                               edgecolor=teacher_color, linewidth=1.7, linestyle="--"))
        axis.plot(box[0], box[1], marker="+", color=teacher_color, markersize=5)
    for box, score, label in zip(decoded["boxes"], decoded["scores"], decoded["labels"]):
        axis.add_patch(Polygon(footprint(box), closed=True, fill=False,
                               edgecolor=prediction_color, linewidth=1.6))
        axis.plot(box[0], box[1], marker="o", color=prediction_color, markersize=3)
        name = NAMES[int(label)] if 0 <= label < len(NAMES) else str(label)
        axis.annotate(f"{name} {score:.2f}", (box[0], box[1]),
                      xytext=(6, 5), textcoords="offset points", fontsize=7,
                      color=prediction_color,
                      bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1))
    axis.legend(handles=[
        Line2D([], [], color=prediction_color, label=f"New head ({len(decoded['boxes'])})"),
        Line2D([], [], color=teacher_color, linestyle="--",
               label=f"Teacher pseudo-labels ({len(teachers)})"),
    ], loc="upper right", fontsize=7, framealpha=.95)

    weights_hash = file_hash(args.weights)
    figure.suptitle(f"Spatial middle fusion: real cached frame {frame}\n"
                    f"{args.frame.parent.name} recording, all panels in the moving velodyne frame",
                    fontsize=15, color="#0f172a")
    figure.supxlabel(
        "Feature magnitude is not semantic confidence. RGB gates are mixing weights, not validated sensor reliability.\n"
        "The heatmap and orange boxes come from the new head; blue boxes are detector-generated training references, not ground truth.\n"
        f"CPU cached-feature diagnostic; excludes sensor encoders, tracking and ROS. Checkpoint SHA256: {weights_hash[:16]}",
        fontsize=9, color="#475569",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(json.dumps(dict(output=str(args.output), frame=str(args.frame), frame_index=frame,
                          time=time, device="cpu", threshold=threshold,
                          teacher_boxes=len(teachers), decoded_boxes=len(decoded["boxes"]),
                          checkpoint_sha256=weights_hash, frame_sha256=file_hash(args.frame))))


if __name__ == "__main__":
    main()
