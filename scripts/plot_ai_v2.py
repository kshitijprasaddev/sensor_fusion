#!/usr/bin/env python3
"""Plot measured v2 development results, including the motion regression."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--report", default="results/ai_training_v2.json")
p.add_argument("--output", default="docs/assets/fusion-v2-results.png")
a = p.parse_args()
r = json.loads(Path(a.report).read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1.5, .8, 1.25]})
colors = ["#687b91", "#008773"]
fields = ["lidar_correct_fraction", "radar_exact_correct_fraction", "radar_spatial_set_correct_fraction"]
for offset, (version, color) in enumerate(zip(("v1", "v2"), colors)):
    values = [100 * r["paired_development"][version][key] for key in fields]
    bars = axes[0].bar(np.arange(3) + (offset - .5) * .35, values, .35, label=version, color=color)
    axes[0].bar_label(bars, fmt="%.1f", fontsize=9, padding=3)
axes[0].set(xticks=np.arange(3), xticklabels=["LiDAR exact", "Radar exact", "Radar spatial set"],
            ylim=(0, 105), ylabel="Weak teacher agreement (%)", title="Paired bag 2 development diagnostics")
axes[0].legend(frameon=False, loc="upper center", ncol=2)
motion = [r["paired_development"][v]["velocity_weak_mae_mps"] for v in ("v1", "v2")]
bars = axes[1].bar(["v1", "v2"], motion, color=[colors[0], "#be5839"], width=.55)
axes[1].bar_label(bars, fmt="%.3f", padding=4)
axes[1].set(ylim=(0, .85), ylabel="Weak-target component MAE (m/s)", title="Motion error increased")
history = r["history"]
train_count = r["split"]["train_frames"][1] + 1
selection = r["split"]["selection_frames"]
selection_count = selection[1] - selection[0] + 1
axes[2].plot([v["epoch"] for v in history], [v["train_loss"] / train_count for v in history],
             color=colors[0], label="Training, augmented")
axes[2].plot([v["epoch"] for v in history], [v["selection_loss"] / selection_count for v in history],
             color=colors[1], label="Checkpoint selection")
axes[2].axvline(r["best_epoch"], color="#be5839", linestyle=":", label="Selected epoch")
axes[2].set(xlabel="Epoch", ylabel="Mean weak loss per frame", title="Actual 100-epoch training")
axes[2].legend(frameon=False, fontsize=8)
for axis in axes:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=.15)
    axis.set_axisbelow(True)
fig.suptitle("QueryFusion v2 | Association improved; motion worsened", fontsize=15, fontweight="bold", y=1.02)
fig.text(.5, -.02, "No ground-truth accuracy: approximate calibration, weak labels, previously inspected development recordings.",
         ha="center", fontsize=10, color="#5c6470")
fig.tight_layout(w_pad=2)
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.output, dpi=160, bbox_inches="tight")
