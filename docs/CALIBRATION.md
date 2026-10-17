# Calibration and time

The supplied recordings contain measured extrinsics in `/tf_static`, but no
`sensor_msgs/CameraInfo`, intrinsic calibration file, ego pose, or labeled objects.
The audit in [dataset_audit.json](dataset_audit.json) is extracted from the actual
messages. No calibration values are inferred from the project slides.

## Spatial convention

The fusion reference is `velodyne`: all positions are in meters. A TF transform
whose parent is `velodyne` and child is `cam0` maps camera coordinates **into**
LiDAR coordinates. Invert that transform for projection:

```text
p_camera = inverse(T_lidar_from_camera) * p_lidar
[u, v, 1] ~ K * p_camera
```

Only points with positive optical depth are projected. The existing `cam0`
orientation is used as recorded; an extra optical-frame rotation is not added.
Radar XYZ is transformed with the recorded `velodyne -> radar` TF before matching.
The node reads these transforms from the selected bag instead of hardcoding them.

## Intrinsics are approximate

The default is a 1920 x 1200 rectified pinhole camera with fx = fy = 1100 px and
principal point (960, 600). These values are an explicit assumption. They are
**not a recovered calibration**, and apparent agreement in one image does not
validate them. Both saved images and startup logs show the warning. Association
counts with this configuration are demonstrations, not accuracy measurements.

The recommended import runs inside Docker and writes a fresh configuration:

```bash
./scripts/run.sh calibrate /absolute/path/to/camera.yaml
./scripts/run.sh demo config:=/results/fusion_calibrated.yaml weights:=/models/yolo11n.engine
```

Alternatively, edit `src/group8_fusion/config/fusion.yaml` and rebuild the image:

1. Obtain the rectified projection matrix `P` from the camera calibration. Use its
   left 3 x 3 block for `K` when the image has been rectified with that matrix.
2. Set the actual image width and height. The processor rejects a size mismatch.
3. Set `status: calibrated` and `allow_approximate: false` after verifying it.
4. Check near and far objects, image corners, and reprojection residuals. Record
   the calibration procedure and error on independent targets.

For this already rectified topic, do not apply lens distortion correction again.
If using raw images later, rectify them with the matching camera model first.

## Mixed clocks

Camera and radar headers use Unix-like time. LiDAR and IMU headers use device
uptime. The LiDAR receipt-minus-header offset is approximately 1,775,644,925 s.
The camera also arrives about 77 ms after its own header timestamp in the first
recording. A single constant applied to every sensor would therefore be wrong.

The replay node stamps sensor messages with their **recorded receipt time** and
publishes `/clock` on that same timeline. Offline evaluation uses exactly those
receipt times. It does not change the original bags. Static transforms are
combined and published with transient-local durability so late RViz subscribers
receive the whole frame tree. A time rewind clears synchronization queues and
the Kalman state; track IDs continue increasing within a running process.

The camera anchors each fusion step. The nearest LiDAR and radar sample must lie
within 100 ms; missing LiDAR causes the frame to be skipped and missing radar is
allowed. The live node waits briefly for future samples and keeps bounded queues.
Receipt alignment is a practical playback fallback, not hardware acquisition
synchronization. Small receipt deltas alone do not prove physical simultaneity.

A calibrated system should estimate individual sensor clock offsets and drift
against a common clock or hardware trigger, measure transport delay, and use
acquisition timestamps. Motion compensation additionally requires ego motion.

## Velocity

The radar field `v` is preserved as `radial_velocity_sensor` in the track export.
The bag does not document its sign convention or ego compensation. Consequently
it is not treated as a full planar velocity measurement in the Kalman filter.
Track velocities are relative to the moving LiDAR frame. They are not absolute
road speeds. The IMU is replayed and audited, but is not integrated as odometry
without initialization, biases, and translational motion measurements.
