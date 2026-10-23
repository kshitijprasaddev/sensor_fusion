# Models and algorithms actually used

> Historical classical baseline, validated 17 October 2026. The neural extension
> added 22 October 2026 is documented in [AI pipeline](AI_PIPELINE.md),
> [results](AI_RESULTS.md) and [literature review](LITERATURE_REVIEW.md).
> The current stable profile is [QueryFusion v2](FUSION_V2.md). The separate
> [middle profile](MIDDLE_FUSION.md) adds spatial feature fusion and independent detection.

| Input or stage | Implementation | Learned weights | Output |
| --- | --- | --- | --- |
| Camera | Ultralytics YOLO11n Detect, 640 input, confidence 0.30 | COCO pretrained `yolo11n.pt` | 2D boxes, class labels, confidence |
| Camera acceleration | ONNX opset 17, TensorRT 8.6 FP16, fixed 640 shape | Locally exported `.engine` | Same camera detections |
| LiDAR | ROI filter, 0.25 m voxel reduction, upward ground-plane RANSAC, KD-tree DBSCAN, axis-aligned boxes | None | Geometric 3D obstacles |
| Radar | Finite and ROI filter, recorded TF, 2D DBSCAN at 1.5 m | None | Radar target clusters and median recorded radial velocity |
| Camera/LiDAR association | Project 3D box corners, IoU score, one-to-one Hungarian assignment, IoU >= 0.05 | None | Semantic labels on matched 3D obstacles |
| LiDAR/radar fusion | One-to-one distance matching <= 2.5 m, inverse-variance XY update | None | Fused positions and source provenance |
| Tracking | XY constant-velocity Kalman filter, Mahalanobis + distance gates, Hungarian assignment | None | IDs, relative velocity, covariance, short occlusion predictions |

This is object-level fusion with one deep learning detector. It does not claim
that geometric clustering is a neural network, or that a learned fusion network
was trained on the supplied recordings. Confidence comes from YOLO when matched;
the geometric confidence and measurement variances are heuristic configuration
choices, not empirically calibrated probabilities. Track source labels describe
the latest measurement. Semantic labels may persist through a camera miss.

The radar topic is a point cloud of existing detections, not raw radar ADC data.
Its upstream detector is undocumented. This project does not claim to run CFAR
or recover information absent from those detections.

## Model provenance

`scripts/prepare_model.py` obtains the standard YOLO11n checkpoint through
Ultralytics. It writes SHA-256 hashes for the checkpoint, exported ONNX and
engine into `/models/manifest.json`. Every camera evaluation saves its weight
hash and full configuration. Large models are ignored by Git. Engines must be
built on the deployment GPU and TensorRT version, including on Jetson.

## Alternatives considered

* **YOLO:** the original [YOLO paper](https://arxiv.org/abs/1506.02640) introduced
  single-stage image detection. This implementation uses the smaller, documented
  [YOLO11n](https://docs.ultralytics.com/models/yolo11/) architecture for the 8 GB
  workstation GPU. It uses pretrained weights without task-specific fine-tuning.
* **PointNet++:** [Qi et al.](https://arxiv.org/abs/1706.02413) learn local features
  directly from point sets. A radar or LiDAR segmentation model would need
  suitable task weights and validation on this sensor distribution. It is
  discussed in the project introduction but is not installed or run here.
* **PointPillars:** [Lang et al.](https://arxiv.org/abs/1812.05784) learn a pillar
  representation for 3D detection. KITTI-trained weights are a possible future
  comparison, with careful handling of coordinates, density and domain shift.
  No such detector is represented as evaluated in this project.
* **Tracking:** [SORT](https://arxiv.org/abs/1602.00763) motivates a compact Kalman
  and Hungarian baseline. This project tracks 3D-derived XY measurements, so it
  is an adaptation of that pattern rather than the original 2D SORT code.

## Known failure modes

Clustering can split walls or merge adjacent objects. Axis-aligned boxes do not
estimate yaw. Sparse or isolated radar returns may be noise or may fall below the
minimum cluster size. Unknown objects outside COCO receive geometric labels.
Poor intrinsics and asynchronous motion can cause incorrect semantic matches.
Ego motion appears in relative track velocities. These limitations must be
considered before interpreting a track or a colored box as a true object.
