# Camera, LiDAR and radar processing

This guide describes the implemented QueryFusion v2 path and the diagnostic
viewer added for the November review. The separate experimental spatial head is
explained in [middle fusion](MIDDLE_FUSION.md). Neither path has established
accuracy on an independent labelled test set.

## Why SAM and a GRU appear in the same pipeline

SAM 2 follows an object's pixels and visual identity through camera frames.
Our pipeline uses that visual ID. We do not need a second network merely to
keep the same ID.

The GRU is a motion head. It takes fused features, current and previous 3D
positions, elapsed time and a recurrent hidden state, then predicts relative
horizontal velocity. A camera mask does not by itself measure metric 3D
velocity. Trying a temporal motion model is reasonable, but its benefit remains
unproven here. Compare its velocity error with position differences on the same
independently labelled samples. If motion is not required, or the GRU does not
help, simplify the system.

SAM's memory is described in the [SAM 2 paper](https://arxiv.org/abs/2408.00714).
The GRU is an established recurrent architecture, described by
[Cho et al.](https://arxiv.org/abs/1406.1078); using it is not a novelty claim.

## Camera: boxes prompt masks

YOLO11n uses published COCO weights through a TensorRT FP16 engine, with a
640-pixel input. A qualifying box can initialise a SAM object. Eligible birth
classes are car, truck, bus, person, motorcycle and bicycle. Prompts are box
coordinates, not manually selected points or text prompts.

SAM 2.1 tiny supplies masks, visual IDs and learned object descriptors. It
retains three memory frames and at most eight object slots per replay
sequence. After initialisation, a mask can continue even when a later YOLO
box falls below the threshold. This does not guarantee identity through every
occlusion or correct a wrong initial class.

The review profile uses **0.65**, above the requested 0.6. Run it with
`./scripts/review_demo.sh fusion`. The historical `fusion_ai.yaml` profile and
its published experiment settings remain at 0.30. The existing learned head
was trained with that lower detector cutoff; the review is a changed operating
point, not a newly validated checkpoint.

| Repeated YOLO predictions over 235 images | Cutoff 0.30 | Cutoff 0.65 |
| --- | ---: | ---: |
| Car | 221 | 114 |
| Person | 155 | 146 |
| Skateboard | 115 | 7 |

These are frame-level prediction counts, not unique objects or accuracy.
The cached post-NMS threshold comparison and fresh GPU run at 0.65 agree on
these totals. Higher confidence removes useful boxes too. The review camera
video displays every COCO class above the cutoff, including remaining skateboard
predictions. Skateboard is separately ineligible to initialise SAM.

The meaning of the cutoff is documented in
[Ultralytics prediction settings](https://docs.ultralytics.com/modes/predict/).

## LiDAR: sparse points become 3D proposals

1. Read recorded XYZ and intensity and reject non-finite positions.
2. Normalize intensity by 255 and clip it. Use the current sweep with zero
   time lag.
3. Apply the upstream model's spatial region and voxel preprocessing.
4. Run pretrained VoxelNeXt sparse convolutional features and detection heads.
5. Retain proposals at score 0.25: oriented 3D boxes, classes, scores and
   128 learned features per proposal.

We use published nuScenes weights, not a detector trained on these bags.
Transfer to this rig needs independent 3D labels. This path does not use
DBSCAN to generate the LiDAR boxes. See the
[VoxelNeXt paper](https://arxiv.org/abs/2303.11301).

In the diagnostic viewer, grey is a subsampled preview of raw returns and cyan
is the actual prepared model input. Every displayed box is a LiDAR proposal
before camera association, including false proposals. The processing trace
reports the actual point, voxel and proposal counts for each frame.

## Radar: recorded returns become learned features

Our bag contains detection points, not raw ADC samples or a range-Doppler cube.
We do not implement or claim the radar front end or CFAR. A return supplies XYZ,
a recorded velocity attribute `v`, and radar cross-section `RCS`. Several
returns can come from the same object or background.

The recorded static transform places points in `velodyne`. We reject invalid
positions and keep the forward region: 2 to 65 metres in x, absolute y below
20 metres and absolute z below 3 metres. The nearest 128 eligible returns are
the maximum input to the learned head.

Each return becomes eight values: normalized XYZ, velocity, RCS, projected
image u and v, and image visibility. An MLP embeds them and a contextual
transformer relates the returns. The camera queries attend to these features.
This encoder is trained inside our small fusion head; it is not a separately
pretrained radar object detector.

The viewer shows raw returns in grey and retained returns coloured by the
recorded `v` attribute on a fixed -10 to +10 scale. The sign and motion
convention need checking before interpreting it as object speed. A radar point
count is not a vehicle count.

## How the default fusion combines the evidence

Each SAM object provides an appearance, image box and class query. Learned
attention selects supporting LiDAR proposals and radar returns. Null tokens
allow no match. Association thresholds, duplicate suppression and finite-value
and abrupt-jump checks can reject an observation.

Accepted 3D geometry comes from the selected VoxelNeXt proposal. Radar supports
features and association; the current path does not directly correct the 3D
box position using radar. The GRU predicts relative XY motion. The camera and
LiDAR backbones stay frozen while our small attention and motion head is
trained with weak labels. This is hybrid object-query feature fusion, not
end-to-end supervised training of all sensor backbones or a BEVFusion reproduction.

The common coordinate system is the moving `velodyne` sensor frame, not a
global map. Camera projection is approximate without real intrinsics. Matching
uses bag receipt times with a 100 ms tolerance because device clocks differ;
this does not establish hardware acquisition synchronization.

## Run and record the diagnostic views

Use a graphical terminal on the Ubuntu host. The prepared cache and Docker
images already exist on the project laptop.

```bash
sudo systemctl start docker
sudo setfacl -m u:kshitij0:rw /var/run/docker.sock
cd /home/kshitij0/sensor_fusion
docker info --format '{{.ServerVersion}}'
./scripts/review_demo.sh lidar
```

Close RViz or press Ctrl+C before changing modes. If the launcher terminal is
lost, stop its container from another terminal:

```bash
docker stop group8-review
./scripts/review_demo.sh radar
```

Use `camera` instead of `radar` to inspect cached SAM masks. The diagnostic
modes publish `/review/raw_points`, `/review/processed_points`, `/review/boxes`,
`/review/image`, `/review/metrics`, recorded static TF and `/clock`, in ROS domain
38. Their 10 FPS playback is **cached output replay, not inference speed**.

To run actual fusion with the review cutoff, stop the diagnostic view first:

```bash
./scripts/review_demo.sh fusion
```

That launches the normal neural pipeline in `group8-rviz`, using the review
configuration. Stop it with Ctrl+C or `docker stop group8-rviz`. Its default
domain is 8, as in the existing launcher.

Record from a second host desktop terminal while exactly one RViz window is
visible. Keep it visible and leave its size unchanged:

```bash
cd /home/kshitij0/sensor_fusion
python3 scripts/record_rviz.py --seconds 30
```

The recorder writes a timestamped H.264 MP4 in `results/`. No screen recorder
needs installing on the host; FFmpeg comes from the existing Docker image.

## Rebuild or generate a fresh cache

The helper scripts are mounted read-only into the existing AI image. They need
no rebuild when only the viewer changes. To rebuild the project image:

```bash
./scripts/run.sh ai-build
./scripts/run.sh ai-test
```

The one-time export computes every stage on the GPU:

```bash
./scripts/review_demo.sh prepare
```

It refuses to overwrite a completed or partial export. For another run:

```bash
./scripts/review_demo.sh prepare --output /results/review_new
./scripts/review_demo.sh lidar --cache /results/review_new
```

The manifest records the effective configuration, configuration and fusion
checkpoint hashes, prediction counts and per-frame processing evidence. These
outputs and full videos stay in ignored `results/`. The original bags and
upstream weights stay local too.

## What the evidence supports

The review GPU export processed all 235 frames of the first bag. Four H.264
videos show YOLO, SAM, LiDAR and radar. The supplied two related bags contain
397 camera images in total, about 40 seconds. Only 164 frames train the small
heads, and both bags have been used during development. We need a new untouched
recording, real calibration and independent labels to establish generalisation,
association accuracy or motion benefit. A smooth RViz recording is useful
engineering evidence, not proof of production readiness.
