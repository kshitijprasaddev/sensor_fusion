# Understand the project

This guide explains what we built and what you can honestly tell the supervisor.
For exact measurements, read [results](AI_RESULTS.md) and [v2 experiments](FUSION_V2.md).
For terminal instructions, use the [run guide](RUN_GUIDE.md).
For the current mask/point display and runtime safeguards, read
[display and robustness](DISPLAY_AND_ROBUSTNESS.md).

## Have we achieved a modern AI solution?

**Yes, we have implemented a working deep learning research prototype. We have
not established state-of-the-art accuracy or finished its scientific validation.**

The original version used YOLO for images, geometric point clustering and a
classical tracker. The neural version also learns LiDAR detection, visual
identity propagation, sensor association and relative motion. Those models run
on the actual bags, inside Docker, and publish results to ROS 2 and RViz.

Using a newer paper does not automatically produce a better system. A strong
project explains its design, compares versions fairly, publishes negative
results and states what evidence is still missing. DBSCAN, Kalman filters and
Hungarian assignment are useful baselines. Their age alone does not establish
that a replacement performs better.

## The problem in one example

Imagine a car in front of the sensors. The image shows its appearance. LiDAR
measures the shape and distance of surfaces. Radar supplies sparse returns with
range-related geometry, a recorded velocity attribute and reflection strength.

We want one object record with a consistent ID, a class, a 3D box, a relative
motion estimate and the sensors that supported the observation. Three separate
sensor detections do not automatically mean three different cars.

## What happens to one frame?

1. **Read the recording.** The replay node reads the supplied ROS bag without
   modifying it. It publishes simulated time using `/clock`.
2. **Choose nearby measurements.** Camera images drive processing. We choose
   nearby LiDAR and radar messages using bag receipt times, with a 100 ms
   tolerance. The LiDAR device clock differs from the image/radar header clocks,
   so header times cannot simply be compared as recorded. Receipt-time matching
   is an explicit approximation, not hardware synchronization.
3. **Put geometry in one coordinate system.** Recorded static transforms place
   sensor measurements in the `velodyne` frame. This lets us compare positions
   in meters. Image projection additionally needs camera intrinsics.
4. **Find image objects.** YOLO11n proposes boxes and classes such as car or
   person. It provides prompts for new visual tracks.
5. **Keep visual identities.** SAM 2.1 tiny segments prompted instances and uses
   transformer memory to carry masks and IDs across frames. We retain three
   memory frames and at most eight object slots per replay sequence.
6. **Detect 3D objects.** VoxelNeXt processes LiDAR XYZ and recorded intensity
   through a pretrained sparse 3D convolutional network. It predicts oriented
   3D boxes, classes, scores and learned features. It replaces DBSCAN detection
   in the neural profile.
7. **Represent radar returns.** We filter invalid/out-of-range returns and encode
   XYZ, recorded velocity, RCS, image position and visibility. RCS means radar
   cross section, a reflection-strength attribute. We do not claim a separately
   pretrained radar detector.
8. **Learn which evidence belongs together.** QueryFusion compares each visual
   object's descriptor and geometry with LiDAR and radar features using
   attention. An explicit null token means "no suitable observation." Version 2
   adds contextual radar self-attention, a learned camera descriptor gate,
   descriptor dropout and spatially distributed weak radar targets.
9. **Estimate relative motion.** A trained GRU keeps a small recurrent state per
   identity and predicts XY velocity from fused features and position history.
   Without validated ego motion this is not absolute road speed.
10. **Publish and display.** ROS publishes objects, IDs, masks, boxes, motion
    arrows and timing information. The default RViz view shows camera masks,
    colored LiDAR object points, IDs and distance guides. Boxes and motion
    vectors are optional. RViz is a viewer of
    recorded sensor playback, not a physics simulator or an accuracy test.

## Which parts were trained by us?

| Part | What it does | Where the learned weights came from |
| --- | --- | --- |
| YOLO11n | Image proposals and class labels | Published pretrained model |
| VoxelNeXt | LiDAR 3D detection and sparse features | Authors' nuScenes checkpoint |
| SAM 2.1 tiny | Masks and temporal visual memory | Published checkpoint through pinned Ultralytics |
| QueryFusion v1 and v2 | Association of object queries and sensor evidence | Trained locally on frozen features from bag 1 |
| GRU motion head | Relative XY motion | Trained locally on weak position differences |
| Radar MLP and v2 contextual transformer | Radar features for fusion | Trained as part of QueryFusion, not a standalone radar detector |

We did not train the large perception backbones from scratch. They stay frozen.
We did not reproduce BEVFusion, RaCFormer, RAF or every paper in the literature
review. Those names describe research comparisons where explicitly marked.

## Is this early, middle or late fusion?

It is **hybrid object-query feature fusion**. We already have object proposals,
but we combine learned internal features before final association and motion
prediction. This goes beyond matching only finished object positions.

It is reasonable to explain it as a form of middle-level feature fusion with
object queries. It is not dense raw-sensor fusion or a single detector trained
end to end from pixels and point clouds. Feature fusion and temporal tracking
are related stages, not interchangeable terms.

## Are all objects in a global frame? Do we use tf2?

We have recorded static sensor transforms and publish a TF tree that RViz and
tf2 consumers can use. The processing code also applies homogeneous transform
matrices derived from those recorded transforms.

The common frame is **`velodyne`, attached to the moving sensor rig**. It is not
a stationary world map. We do not have a validated `map`/`odom` trajectory, global
localization or ego-motion compensation. An object moving in this frame may
reflect both the object's motion and the vehicle's motion.

## What does a detection count mean?

`camera_detections: 2` means two image proposals for that processed frame.
`lidar_objects: 6` means six VoxelNeXt box proposals. In the neural profile,
`radar_objects: 50` means 50 filtered radar returns, not 50 vehicles.

`camera_matches`, `tracks` and `three_sensor_objects` count accepted fused
observations. Summing them across frames counts repeated observations of the
same object. It does not give the number of distinct physical objects or an
accuracy percentage. The metrics identify `pipeline` and `fusion_schema` so
versions can be distinguished.

## What is still holding us back?

The supplied data has 397 images across two short, related recordings. We lack
real camera intrinsics and independently labelled 3D boxes/identities. A matrix
with assumed focal length is usable for a clearly labelled demo, but it is not
calibration. Our weak training labels inherit errors from that assumption.

The exported box center and dimensions come from the selected VoxelNeXt box.
Radar contributes association and motion features; it does not independently
correct the 3D box. The neural system needs a visible camera instance and a
LiDAR association, so radar-only and off-camera objects are outside its current
tracking scope. SAM masks can drift, and eight lifetime slots limit births.

The first neural release took about 0.37 seconds per camera frame on the RTX
4060, before video encoding and ROS transport. It cannot keep up with the
recording's 10 Hz camera at full speed. We play it slowly for the demo.

## What would make the project stronger?

| Next evidence or experiment | Why it matters | What would count as success |
| --- | --- | --- |
| Real intrinsics and verified timing/extrinsics | Prevents learning projection and clock errors | Independent reprojection and timestamp checks |
| More varied scenes with independent 3D boxes and IDs | Separates teacher imitation from accuracy | Detection and tracking metrics on untouched scenes |
| Controlled v1/v2 and baseline comparisons | Tests whether each neural change helps | Same inputs, split rules, metrics and hardware |
| True sensor failures and adverse conditions | Tests whether fusion survives missing evidence | Accuracy/coverage and latency under documented perturbations |
| Ego pose and motion compensation | Gives a stable frame for trajectories | Verified world-frame positions and velocity evaluation |
| Better birth/death handling and deployment optimization | Improves coverage and processing rate | ID metrics, capacity checks and measured full-rate performance |

See [literature review](LITERATURE_REVIEW.md) for modern directions and
[v2 experiments](FUSION_V2.md) for the work actually implemented. A new bag kept
unseen until the next design is frozen is needed for a fresh test.

## A clear explanation to give the supervisor

> We replaced geometric obstacle clustering and classical identity tracking in
> our neural profile with pretrained LiDAR detection, SAM temporal memory and a
> locally trained attention/GRU fusion head. We implemented a second fusion head
> to address camera-feature overfitting and ambiguous radar supervision. We
> evaluate execution, teacher agreement and missing-input behavior separately.
> Real calibration, independent labels and unseen scenes are the next steps
> before we can make accuracy or state-of-the-art claims.
