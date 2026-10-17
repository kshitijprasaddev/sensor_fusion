# Presentation and local startup

The repository is `/home/kshitij0/sensor_fusion`. The raw bags remain in
`/home/kshitij0/Downloads/studentProject1`. Docker mounts the bags read-only.
Models are in `models/`, generated outputs are in `results/`, and the report is
`docs/PROJECT_REPORT.md`. The existing Docker image already includes ROS Humble,
TensorRT, the processing pipeline and RViz2. Rebuilding is unnecessary to present.

## Start the live demo

Open a terminal on the Ubuntu graphical desktop:

```bash
sudo systemctl start docker
sudo setfacl -m u:kshitij0:rw /var/run/docker.sock
cd /home/kshitij0/sensor_fusion
./scripts/run.sh demo weights:=/models/yolo11n.engine
```

Wait about 8 seconds for replay to start. The first bag loops at half speed.
A 23.6-second recording takes about 47.2 seconds to replay. Half speed gives the
processing and graphics enough time; it is not a claim of full-rate performance.

To stop, press Ctrl+C in the launch terminal. If the original terminal is lost:

```bash
docker stop group8-rviz
```

To show the second bag after stopping the first:

```bash
cd /home/kshitij0/sensor_fusion
./scripts/run.sh demo bag:=/data/studentProject1 weights:=/models/yolo11n.engine
```

The second bag lasts about 32.8 seconds at half speed. Add `loop:=false` for a
single pass. RViz remains open at the final frame until you stop the launch.
If a previous demo is still running, stop it before starting another. Socket
access can reset when Docker restarts, so repeat the `setfacl` command if access
is denied. Do not run the helper itself with sudo, since that changes display
credentials and ownership of generated files.

## Saved full recording

The full RViz2 window recording is [rviz-full-demo.mp4](videos/rviz-full-demo.mp4).
The video is 98 seconds long, 1600 × 1000, H.264, 15 fps, and about 37 MB.
The second bag starts at 00:56. It contains both bags in order, with playback
at half speed, including startup
and the final held frame. It is a real screen recording with no audio. The first
section uses `studentProject` and the second uses `studentProject1`.

To play it locally:

```bash
xdg-open /home/kshitij0/sensor_fusion/docs/videos/rviz-full-demo.mp4
```

## Record RViz again

Keep RViz visible on the desktop. In a second terminal:

```bash
cd /home/kshitij0/sensor_fusion
python3 scripts/record_rviz.py --seconds 60 --output results/my-presentation.mp4
```

This records the RViz window at 15 video frames per second, without audio. It uses
FFmpeg already installed in the Docker image. The actual sensor updates are
slower than the video frame rate. Avoid covering, minimizing or resizing RViz
while recording. X11 window capture may also capture overlapping windows. The
script refuses to overwrite an existing video; choose a new output name.

## Presentation walkthrough

1. Explain that this is replay of real recorded data, not a synthetic simulation.
2. Show the camera panel: YOLO11n supplies object classes and image boxes.
3. Show the 3D view: blue/cyan LiDAR points, orange radar targets, green fused boxes.
4. Explain that transforms put sensors in a common frame. Clustering finds
   geometric objects, then association combines observations of the same target.
5. Point to object IDs and motion arrows: a constant-velocity Kalman filter and
   Hungarian assignment track targets over successive frames.
6. State the limitations: camera intrinsics are approximate, timing uses bag
   receipt times, velocity is relative to the moving sensor, and there are no
   ground-truth labels for accuracy measurement.
7. Show validation: 397 camera frames processed across both bags, 20 tests passed
   inside the container, and numerical CI passed on GitHub.

Use the saved video as a fallback if the live display fails. Check the live demo
on the presentation laptop before the session. The TensorRT engine is specific
to the tested GPU/software setup and should be rebuilt when that changes.
