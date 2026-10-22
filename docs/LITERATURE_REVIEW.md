# Learned multi-sensor perception and tracking

Review date: 22 October 2026. This is a targeted implementation-oriented review,
not an exhaustive survey or a claim to reproduce every recent paper. Primary
papers and official repositories are linked below. Published benchmark results
are not measurements on our recordings.

## Research question and original baseline

Can pretrained neural representations and learned cross-modal association replace
our hand-designed obstacle clustering, position matching and motion tracker on
one camera, one LiDAR and recorded radar detections, using an 8 GB RTX laptop?

The original YOLO11n + DBSCAN + Kalman/Hungarian implementation remains a
classical baseline. Those algorithms are useful comparison tools, but the
original pipeline did not learn LiDAR detection, radar association or temporal
visual identity. The mentor's request therefore changes the experimental scope.
Algorithm age alone is not an accuracy criterion; comparisons need labels,
matched inputs and measured compute. Neural methods also contain non-learned
operations such as projection, suppression and coordinate transforms.

## LiDAR representation and detection

**VoxelNeXt, CVPR 2023.** Chen et al. use sparse voxel processing for 3D detection
and tracking. Its released nuScenes detector checkpoint provides learned 3D
boxes, semantic classes and sparse intermediate features. We use the original
checkpoint, not a re-created randomly initialized model. [Paper](https://arxiv.org/abs/2303.11301),
[official implementation and model downloads](https://github.com/JIA-Lab-research/VoxelNeXt).

Our adapter preserves XYZ, normalizes recorded intensity by 255, and adds zero
sweep-time lag because these inputs are single sweeps. It does not invent the
historical sweeps used in the upstream setting. Checkpoint/data differences,
intensity scaling and coordinate assumptions create domain-shift risk. We keep
oriented boxes and pool 128-dimensional sparse features at their centers. This
replaces geometric obstacle clustering in the neural profile.

**Point Transformer V3, CVPR 2024.** Wu et al. emphasize scalable point-cloud
representations and efficient serialized attention. This is a strong alternative
backbone, but a backbone is not by itself a trained 3D detection head for this
sensor. We do not represent it as installed or evaluated. [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.html),
[official implementation](https://github.com/Pointcept/PointTransformerV3).

Selection: VoxelNeXt has a directly usable released detection checkpoint and
fits the existing CUDA/PyTorch deployment. Its 2023 date is explicitly retained;
it is not advertised as the newest LiDAR architecture or current state of the art.

## Camera/LiDAR and radar feature fusion

**TransFusion, CVPR 2022.** Bai et al. introduce transformer-based LiDAR-camera
fusion with attention that selects image evidence for object queries. This
motivates attention-based association, but our compact head does not reproduce
TransFusion's architecture or training. [Paper](https://arxiv.org/abs/2203.11496).

**BEVFusion, ICRA 2023.** Liu et al. place modality features in a shared bird's-eye
view representation. It is relevant to a future calibrated dense feature-fusion
system. Our system uses object-query features instead of reproducing dense BEV
lifting. The original implementation is archived and has a different dependency
stack. [Paper](https://arxiv.org/abs/2205.13542),
[official implementation](https://github.com/mit-han-lab/bevfusion).

**LiRaFusion, 2024.** Song et al. combine LiDAR and radar through joint voxel
features and adaptive feature-map fusion. This supports encoding radar evidence
rather than treating its clusters as final objects. The paper's nuScenes
training/evaluation cannot be transferred as an accuracy claim to our short bags.
[Paper](https://arxiv.org/abs/2402.11735).

**RCBEVDet, CVPR 2024, and RCBEVDet++.** Lin et al. encode radar geometry and RCS
and fuse radar/camera representations through attention. RCS is therefore a
reasonable input channel to examine. Our point-token MLP is much smaller and is
not RadarBEVNet or an implementation of RCBEVDet. [RCBEVDet](https://arxiv.org/abs/2403.16440),
[RCBEVDet++](https://arxiv.org/abs/2409.04979).

**RaCFormer, CVPR 2025.** Chu et al. study radar-camera query fusion and the
problem of inaccurate image-to-BEV alignment. This directly reinforces the
importance of our missing calibration and timing limitations. We review the
method without claiming its benchmark results or weights were evaluated here.
[Paper](https://arxiv.org/abs/2412.12725),
[official implementation](https://github.com/cxmomo/RaCFormer).
The released setup uses the MMDetection3D ecosystem, calibrated nuScenes inputs
and a multi-GPU training recipe. Our one-camera bags are not a drop-in equivalent
to that training setup. We have not installed or executed RaCFormer.

**CenterFusion, WACV 2021.** Nabati and Qi combine camera predictions and radar
feature maps to refine object properties. Although older, it illustrates that
"middle fusion" can combine proposals and learned features without fusing every
raw measurement at the earliest stage. [Paper](https://arxiv.org/abs/2011.04841).

Selection: the implemented **project-specific QueryFusion** uses SAM object
pointers, VoxelNeXt sparse features, normalized geometry and radar XYZ/velocity/RCS
tokens. Two learned cross-attention blocks include explicit null-observation
tokens. It is hybrid feature-and-object fusion with frozen pretrained encoders,
not an end-to-end trained BEV detector and not a reproduction of the above papers.
No pretrained tri-modal checkpoint is claimed for this custom module.

## Recent reliability research and the second implementation

**MoME, CVPR 2025.** Park et al. route object queries among camera, LiDAR and
joint expert decoders according to sensor feature quality. This targets the
failure of fused detectors when a modality becomes unreliable.
[Paper](https://arxiv.org/abs/2503.19776). Our v2 uses descriptor dropout and a
camera descriptor gate, but does not reproduce MoME's three expert decoders.
This is a conceptual comparison, not an evaluated MoME implementation.

**RAF, ECCV 2026 according to the authors' arXiv record.** Park, Jeong and Yoon
supervise pixel reliability while integrating camera evidence with frozen
LiDAR/radar backbones for adverse weather. This supports examining reliability
and selective fusion rather than just increasing model size.
[Primary paper record](https://arxiv.org/abs/2607.04587). We do not have the
reliability labels or weather evaluation used there, and do not claim to
implement RAF. Its published results are not our measurements.

**Confidence calibration.** Guo et al. study temperature scaling for neural
probabilities. [Paper](https://arxiv.org/abs/1706.04599). V2 searches a fixed small
temperature grid on a separate segment of bag 1 after checkpoint selection.
These temperatures fit noisy pseudo-labels, not physical correctness. A tiny
calibration segment cannot guarantee trustworthy confidence on other scenes.

The implemented v2 adds radar return self-attention before cross-modal attention,
separate descriptor/geometry encoders, a learned camera descriptor gate and
descriptor/missing-radar augmentation. Its weak radar loss distributes target
mass over nearby plausible returns rather than forcing a single nearest return.
The geometric radius remains a disclosed hand-set assumption. This is an
original compact engineering experiment, not a new state-of-the-art paper.
See [paired measurements and limitations](FUSION_V2.md).

## Temporal identity and neural motion

**SAM 2, 2024.** Ravi et al. introduce promptable image/video segmentation with
transformer memory. We use the SAM 2.1 tiny checkpoint through the pinned
Ultralytics implementation, prompt births with YOLO and propagate mask identities
with neural memory. This tracks visual mask instances; it is not a pretrained
3D multi-object tracker. The ROS adapter attaches learned 3D associations to those
visual identities. [Paper](https://arxiv.org/abs/2408.00714),
[official project](https://github.com/facebookresearch/sam2).

The implementation maintains a bounded FIFO memory, explicitly limits object
slots and reports rejected birth prompts. New-object suppression is geometric.
There is no Kalman filter or Hungarian solver in the neural runtime. A trained
GRU estimates relative XY velocity from fused features and position history;
it is a separate project-specific motion head, not part of SAM's checkpoint.

**CoTracker3, ICCV 2025.** Karaev et al. learn point trajectories and use
pseudo-labeled real videos in training. This makes weak supervision relevant,
but tracking points is different from detecting and tracking complete 3D objects.
It is reviewed as an alternative, not silently counted as implemented.
[Paper](https://arxiv.org/abs/2410.11831),
[official implementation](https://github.com/facebookresearch/co-tracker).

**TrackFormer, CVPR 2022, and MOTIP, CVPR 2025.** TrackFormer propagates track
queries through attention. MOTIP predicts identities from trajectory context.
These show alternatives to a handcrafted motion/assignment pipeline. Their
training data, outputs and deployment requirements differ from our chosen
mask-memory tracker. Neither is represented as reproduced. [TrackFormer](https://arxiv.org/abs/2101.02702),
[MOTIP](https://openaccess.thecvf.com/content/CVPR2025/papers/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.pdf).

**FutrTrack, 2025 preprint.** Teye et al. combine camera/LiDAR representations
with transformer refinement and tracking. It is a relevant future full 3D
tracking comparison once calibration, labels and compatible detector features
are available. [Paper](https://arxiv.org/abs/2510.19981).

## Supervision, evaluation and limitations

The project has 397 camera frames, no ground-truth 3D boxes/IDs, approximate
camera intrinsics and no validated ego pose. Training a large fusion model from
scratch on this data would not establish generalization. We freeze published
encoders and train only a compact association/motion head.

Weak LiDAR targets come from compatible-class projected-box overlap. Weak radar
targets come from nearby points relative to that selected LiDAR box. Motion
targets are clipped finite differences of weak 3D observations with the same SAM
identity. These are explicitly fallible pseudo-labels. There is no supervision
from manually annotated accuracy labels.

Both versions use the first 70% of bag 1 for training and the next 10% as a purge
gap. V1 used the final 20% for checkpoint selection. V2 divides that final
segment into 10% selection and 10% temperature fitting. Bag 2 receives no
gradient updates or checkpoint selection, but its v1 results informed the v2
design. Therefore v2 reports it as a **development diagnostic**, not an untouched
test. Both recordings are short and visually related. A fresh independently
annotated recording is required for a new-scene generalization test.

Report association agreement with weak labels, null selections, feature
ablations, processing latency, memory use and ROS delivery. Do not call these
mAP, HOTA, AMOTA, identity-switch accuracy or calibrated uncertainty. Fixed
reporting covariances and confidence thresholds are heuristic. True accuracy
requires independent labels and real calibration. Sensor receipt-time matching
and moving-frame relative velocities remain unchanged limitations.

## Implementation status

| Component | Status | Training or origin |
| --- | --- | --- |
| YOLO11n | Executed camera proposals | Published COCO pretrained weights |
| VoxelNeXt | Executed LiDAR detector/features | Authors' nuScenes checkpoint |
| SAM 2.1 tiny | Executed temporal mask identities | Published checkpoint via Ultralytics |
| QueryFusion v1/v2 attention | Implemented and locally trained | Weak associations from first bag only |
| GRU relative velocity | Implemented and locally trained | Weak position differences; no ego compensation |
| BEVFusion, TransFusion, LiRaFusion, RCBEVDet, RaCFormer | Literature comparisons | Not installed or claimed evaluated |
| PTv3, CoTracker3, TrackFormer, MOTIP, FutrTrack | Literature alternatives | Not installed or claimed evaluated |
| MoME and RAF | Reliability research comparisons | Not installed or claimed evaluated |

See [AI implementation and commands](AI_PIPELINE.md) and the machine-readable
validation evidence for measured execution results and checkpoint provenance.
