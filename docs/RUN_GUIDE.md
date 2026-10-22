# Terminal commands on the project laptop

Run these in an Ubuntu graphical terminal on the host. The project is at
`/home/kshitij0/sensor_fusion`. ROS Humble and the ML dependencies run inside
Docker. The host can remain Ubuntu 24.04; the container supplies Ubuntu 22.04.

## 1. Start Docker and allow this account to use it

```bash
sudo systemctl start docker
sudo setfacl -m u:kshitij0:rw /var/run/docker.sock
cd /home/kshitij0/sensor_fusion
docker info --format '{{.ServerVersion}}'
nvidia-smi
```

The socket permission can reset after a Docker restart. Repeat the `setfacl`
command if Docker reports permission denied. Docker access gives powerful host
access; these commands assume your own project workstation.

## 2. Launch the current neural RViz demo

```bash
cd /home/kshitij0/sensor_fusion
./scripts/run.sh ai-demo
```

This uses `group8-sensor-fusion:ai`, `fusion_ai.yaml` and the v2 checkpoint. The
container is named **`group8-rviz`**. The recording plays at 0.1 speed and loops.
Slow playback gives the neural models time to process. It is not a measurement
of full-rate performance.

For the second bag, stop the first demo and run:

```bash
./scripts/run.sh ai-demo bag:=/data/studentProject1
```

For one pass without looping:

```bash
./scripts/run.sh ai-demo loop:=false
```

## 3. Inspect it from a second terminal

These commands require the demo container to be running:

```bash
docker ps --filter name=group8-rviz
docker exec -it group8-rviz bash
source /opt/ros/humble/setup.bash
source /workspace/sensor_fusion/install/setup.bash
ros2 topic list
ros2 topic echo /fusion/metrics --once
ros2 topic echo /fusion/tracks --once
ros2 run tf2_ros tf2_echo velodyne radar
```

Press Ctrl+C to stop `tf2_echo`, then `exit` to leave this shell. The prompt
`I have no name!` can appear because the container runs with your numeric host
UID. It does not mean ROS failed. `group9-rviz` is not our container name.

Inspect `fusion_schema: group8-query-fusion-v2` in the metrics to confirm v2.

## 4. Stop playback and RViz

Press Ctrl+C in the launch terminal, or run this on the host:

```bash
docker stop group8-rviz
```

The helper removes the stopped container. Models, results and the raw data
remain in their mounted host directories. A new launch creates a new container.

## 5. Evaluate without opening RViz

```bash
cd /home/kshitij0/sensor_fusion
./scripts/run.sh ai-test
./scripts/run.sh ai-evaluate /data/studentProject1 \
  --weights /models/yolo11n.engine \
  --output /results/my_v2_evaluation --video
```

Output appears under `~/sensor_fusion/results/my_v2_evaluation/`, including
`summary.json`, `frames.csv`, `tracks.json` and an annotated `fusion.mp4`.
This MP4 is a camera overlay, not a recording of the whole RViz window.

Run the full checks and both complete bags with:

```bash
./scripts/run.sh ai-validate /results/my_v2_validation
```

Use a fresh output name for each experiment to preserve earlier evidence.
Full validation takes several minutes and uses the GPU.

## 6. Compare the versions and modes

| Command or configuration | Purpose |
| --- | --- |
| `ai-demo` | Current neural v2: all three modalities |
| `ai-demo config:=.../fusion_ai_v1.yaml` | Preserved neural v1 |
| `demo` | Original classical baseline |
| Baseline `--modality lidar` | LiDAR-only geometric baseline |
| Baseline `--modality camera_lidar` | Camera and LiDAR baseline |
| Baseline `--modality lidar_radar` | LiDAR and radar baseline |
| Baseline `--modality all` | Original three-sensor baseline |

```bash
# Preserved neural v1 in RViz
./scripts/run.sh ai-demo \
  config:=/workspace/sensor_fusion/src/group8_fusion/config/fusion_ai_v1.yaml

# Classical baseline in RViz
./scripts/run.sh demo weights:=/models/yolo11n.engine

# Example baseline ablation, without RViz
./scripts/run.sh evaluate /data/studentProject \
  --modality lidar --output /results/my_classical_lidar
```

Stop each RViz demo before starting another. Neural evaluation accepts `all`;
the classical switches are not neural sensor-removal experiments. V2's separate
training report contains descriptor perturbations and missing-token diagnostics.

## 7. Rebuild after changing source or pulling updates

```bash
cd /home/kshitij0/sensor_fusion
git pull --ff-only
./scripts/run.sh ai-build
./scripts/run.sh ai-test
./scripts/run.sh ai-demo
```

The Docker image contains a copy of the source. Editing host source does not
update that image until you rebuild it. Models and results are mounted at
runtime, so they persist across image rebuilds.

For a fresh workstation with the NVIDIA driver, Container Toolkit and data ready:

```bash
./scripts/run.sh build
./scripts/run.sh prepare --engine
./scripts/run.sh ai-build
./scripts/run.sh ai-prepare
./scripts/run.sh ai-test
./scripts/run.sh ai-demo
```

The base is the requested `nvcr.io/nvidia/tensorrt:23.08-py3`. The neural extension
adds sparse CUDA operators. It was built for this RTX 4060's architecture, not
validated on Jetson. See [environment](ENVIRONMENT.md) for the exact stack.

## 8. Reproduce v2 training

The small trained heads are included in Git. Retraining is optional for the demo.
The large published weights and feature caches are downloaded/generated locally.

```bash
./scripts/run.sh ai-extract --output /results/features_reproduced
./scripts/run.sh ai-train \
  --features /results/features_reproduced \
  --output /models/query_fusion_v2_reproduced.pt \
  --report /results/training_v2_reproduced.json
```

This creates a separate checkpoint. To use it, copy `fusion_ai.yaml` to a new
configuration and change `neural.fusion.weights` to that `/models/...` path.
Pass the new configuration explicitly to the demo or evaluation. Training does
not silently switch a running container's model.

`ai-train-v1` preserves the original v1 training procedure. All custom training
uses weak labels. Re-extract and retrain after importing actual calibration;
the checkpoint rejects changed camera intrinsics.

## 9. Record the whole RViz window

Keep the neural demo open and visible. In another host terminal:

```bash
cd /home/kshitij0/sensor_fusion
python3 scripts/record_rviz.py --seconds 60 --output results/my_rviz_recording.mp4
```

Keep the window size unchanged during recording. Use a fresh output filename;
the script refuses overwrites. Full videos stay local under `results/`. The
GitHub README uses a smaller GIF for browser preview. Recording increases GPU
and CPU load, so do not use recording timings as a controlled speed benchmark.

## Where everything lives

| Host path | Container path or meaning |
| --- | --- |
| `/home/kshitij0/sensor_fusion` | Source copied into `/workspace/sensor_fusion` at build |
| `/home/kshitij0/Downloads/studentProject1` | Read-only `/data` containing both bags |
| `/home/kshitij0/sensor_fusion/models` | `/models`, published and locally trained weights |
| `/home/kshitij0/sensor_fusion/results` | `/results`, local evaluation, videos and logs |
| `docs/validation` | Selected measured evidence published on GitHub |

If the data moves, set `DATA_DIR=/absolute/new/path` before the helper command.
Neither raw bags nor large model downloads are uploaded to GitHub.
