#!/usr/bin/env python3
"""Plot measured weak-supervision training and feature diagnostics."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--training',default='/results/ai_training.json')
p.add_argument('--output',default='/results/ai_training.png')
a=p.parse_args();r=json.loads(Path(a.training).read_text());history=r['history']
train_count=r['split']['train_frames'][1]-r['split']['train_frames'][0]+1
val_count=r['split']['validation_frames'][1]-r['split']['validation_frames'][0]+1
fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
axes[0].plot([v['epoch'] for v in history],[v['train_loss']/train_count for v in history],label='Training')
axes[0].plot([v['epoch'] for v in history],[v['validation_loss']/val_count for v in history],label='Validation')
axes[0].axvline(r['best_epoch'],color='#666666',linestyle=':',label='Selected checkpoint')
axes[0].set(xlabel='Epoch',ylabel='Mean weak-supervision loss per frame',title='Training on bag 1 only')
axes[0].legend(frameon=False);axes[0].grid(alpha=.2)
labels=['Full head','Zero camera\nfeatures','Zero LiDAR\nfeatures','Zero radar\nfeatures']
rows=[r['held_out']]+list(r['feature_ablations'].values());x=np.arange(len(rows));w=.36
axes[1].bar(x-w/2,[100*v['lidar_teacher_agreement'] for v in rows],w,label='LiDAR association',color='#247ba0')
axes[1].bar(x+w/2,[100*v['radar_teacher_agreement'] for v in rows],w,label='Exact radar-point association',color='#e08736')
axes[1].set_xticks(x,labels,fontsize=8)
axes[1].set(ylabel='Agreement with weak targets (%)',ylim=(0,100),title='Bag 2: feature-zeroing diagnostics')
axes[1].legend(frameon=False,fontsize=8);axes[1].grid(axis='y',alpha=.2)
fig.suptitle('Pseudo-label diagnostics, not ground-truth detection or tracking accuracy',fontsize=12)
Path(a.output).parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=180)
