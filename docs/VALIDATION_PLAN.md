# Validation protocol

## Completed checks and evidence

Actual measurements are stored in `docs/validation` and described in the project
report. Raw logs, annotated frames, videos and per-frame exports remain under
`results`, which is excluded from Git. A summary is never labeled as accuracy
merely because objects were detected or messages were published.

## Reproduce

After building and preparing the engine, `./scripts/run.sh validate` repeats the
numerical tests, dependency check, environment capture, dataset audit, backend
comparison and all four sensor ablations on both recordings. It writes its
evidence under `results/validation`. The individual commands are:

```bash
./scripts/run.sh build
./scripts/run.sh test
./scripts/run.sh prepare --engine
./scripts/run.sh audit
./scripts/run.sh evaluate /data/studentProject --weights /models/yolo11n.engine --output /results/studentProject --video
./scripts/run.sh evaluate /data/studentProject1 --weights /models/yolo11n.engine --output /results/studentProject1 --video
```

Run each ablation on the identical receipt-time matches, configuration and camera
anchor frames using `--modality lidar`, `--modality camera_lidar`, and
`--modality lidar_radar`. Save each output to a different directory. The full
pipeline uses `--modality all`. Latency includes detector, geometric processing,
association and tracking, but excludes deserialization, image annotation, video
encoding and ROS transport. It is not end-to-end sensor latency.

`scripts/verify_ros.py --rviz --min-images 100` subscribes to the live outputs. It requires actual images,
filtered clouds, markers, object messages, metrics, static sensor frames and
confirmed track IDs. It rejects nonfinite object coordinates. This checks the
ROS wiring, not perception accuracy. The `--rviz` option also checks the actual
RViz subscriptions; a separate screenshot verifies the visible result. The
`node_latency_ms` metric includes annotation and publishing, while the offline
latency table measures the shared processing stage only.

## Accuracy evaluation once prerequisites arrive

1. Import and independently verify the real camera intrinsics. Measure temporal
   offsets using a common event or a synchronization target.
2. Annotate held-out camera boxes and 3D tracks. Record class, position, size,
   identity and visibility. Split by recording rather than adjacent frames to
   avoid leakage from almost identical images.
3. Compare camera mAP, 3D/BEV precision and recall, center error, association
   precision, ID switches, and HOTA/IDF1 across the same four ablations.
4. Add odometry and transform tracks into a world frame before reporting absolute
   velocity error. Determine the radar Doppler convention before velocity fusion.
5. Report hardware, sensor timestamps, calibration error, confidence thresholds,
   warmup, median and p95 latency, frame loss, and adverse-condition failures.

Neither supplied bag contains labeled ground truth. The repository therefore
reports execution and association statistics only. No mAP, HOTA, physical speed
accuracy or safety claim is invented from these recordings.
