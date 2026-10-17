#!/usr/bin/env python3
"""Create a measured-results table and plot from offline run summaries."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('runs', nargs='+')
p.add_argument('--output', default='/results/comparison')
args = p.parse_args()
runs = [json.loads((Path(d) / 'summary.json').read_text()) for d in args.runs]
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
bag_names = {'studentProject': 'Bag 1', 'studentProject1': 'Bag 2'}
mode_names = {'lidar': 'L', 'camera_lidar': 'C+L', 'lidar_radar': 'L+R', 'all': 'C+L+R'}
labels = [f"{bag_names.get(r['bag'], r['bag'])}\n{mode_names[r['modality']]}" for r in runs]
fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
fig.patch.set_facecolor('#f5f8fc')
x = np.arange(len(runs))
axes[0].bar(x - .18, [r['latency_ms']['p50'] for r in runs], width=.36, label='Median', color='#147d92')
axes[0].bar(x + .18, [r['latency_ms']['p95'] for r in runs], width=.36, label='95th percentile', color='#213c62')
axes[0].set_ylabel('Processing time per frame (ms)')
axes[0].set_title('Measured processing time')
axes[1].bar(x - .18, [r['totals']['camera_matches'] / r['frames'] for r in runs], width=.36, label='Camera to LiDAR', color='#147d92')
axes[1].bar(x + .18, [r['totals']['radar_matches'] / r['frames'] for r in runs], width=.36, label='Radar to LiDAR', color='#ed9148')
axes[1].set_ylabel('Accepted matches per frame')
axes[1].set_title('Association counts, not accuracy')
for ax in axes:
    ax.set_xticks(x, labels, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
fig.suptitle('Group 8 | Recorded-data validation | Approximate camera intrinsics', fontsize=14, fontweight='bold')
fig.supxlabel('C: camera   L: LiDAR   R: radar     |     Bag 1: studentProject     Bag 2: studentProject1', fontsize=9)
fig.savefig(out / 'comparison.png', dpi=180)
plt.close(fig)
table = ['| Recording | Mode | Frames | Median ms | P95 ms | Camera matches | Radar matches |',
         '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for r in runs:
    table.append(f"| {r['bag']} | {r['modality']} | {r['frames']} | {r['latency_ms']['p50']:.1f} | "
                 f"{r['latency_ms']['p95']:.1f} | {r['totals']['camera_matches']} | {r['totals']['radar_matches']} |")
(out / 'table.md').write_text('\n'.join(table) + '\n')
print('\n'.join(table))
