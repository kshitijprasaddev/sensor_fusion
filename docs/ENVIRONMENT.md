# Reproducible environment

The requested `nvcr.io/nvidia/tensorrt:23.08-py3` image is the base, with ROS 2
Humble Desktop installed inside it. The host may run Ubuntu 24.04; ROS Humble runs
on Ubuntu 22.04 inside Docker. Host ROS is not needed to run the project.

| Component | Container configuration | Reason |
| --- | --- | --- |
| Architecture | x86_64 | This workstation is an RTX PC, not a Jetson |
| Ubuntu / Python | 22.04 / distro Python 3.10 | Supplied by the base image and apt |
| ROS | Humble Desktop, current Jammy apt patch versions at build time | RViz2 and Python message support included |
| CUDA / TensorRT | 12.2.1 / 8.6.1.6 | Preserved from NVIDIA 23.08 |
| cuDNN system library | 8.9.4 | Base container |
| PyTorch / Torchvision | 2.4.0+cu121 / 0.19.0+cu121 | Available upstream x86 GPU wheels; wheel libraries are separate from system TensorRT |
| Ultralytics | 8.4.95 | Requested version |
| NumPy / SciPy | 1.26.4 / 1.11.4 | SciPy 1.8 requires an older NumPy range |
| OpenCV | 4.8.1.78 Python wheel | Ultralytics 8.4.95 requires OpenCV >= 4.6 |
| ONNX / ONNX Runtime GPU | 1.22.0 / 1.23.0 | Requested tooling versions; TensorRT is the tested exported runtime |

The initial course environment mixes Jetson aarch64, an NVIDIA Jetson-specific
PyTorch wheel, TensorRT 8.6, and TensorRT 10/11 references. These are distinct
targets. Installing all of them into this x86 TensorRT 8 image would not produce
a compatible environment. TensorRT 10/11 engines cannot be substituted here.
PyTorch's upstream CUDA wheel also includes its own cuDNN dependency, so its
runtime version differs from the system library used by TensorRT.

Open3D, PyCUDA, pyquaternion and transforms3d are not needed by the implemented
path. NumPy and SciPy handle clouds and transforms; Ultralytics manages TensorRT
bindings. The repository does not claim unused dependencies were tested. ROS
patch versions are resolved from the ROS apt repository; `scripts/environment.py`
records the installed versions in the validation output. This is reproducible
source configuration, not a claim of bit-identical future apt builds.

## GPU and display

Prefer the [official NVIDIA Container Toolkit setup](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The helper uses `--device=nvidia.com/gpu=all` when NVIDIA CDI is available, or
`--gpus all` when an NVIDIA runtime is registered. On this x86
Linux workstation it also supports `GPU_MODE=direct`, exposing `/dev/nvidia0`,
`/dev/nvidiactl`, `/dev/nvidia-uvm` and read-only host driver files. This fallback
uses GPU CUDA inference and Mesa software rendering for RViz. It makes no host
runtime configuration changes and does not require a privileged container.

Use `GPU_MODE=cdi` or `GPU_MODE=toolkit` to force a standard runtime path, or `GPU=0` with
`device:=cpu` / `--device cpu` for CPU inference from the `.pt` checkpoint.
TensorRT engine execution requires CUDA. If `docker` reports permission denied,
grant your user Docker access using your system's administrative procedure.

Display access uses the existing X11 socket and Xauthority cookie. The script
does not disable X server access control with `xhost +`. On a headless machine,
launch `ros2 launch group8_fusion demo.launch.py rviz:=false` in a shell.

## Jetson deployment

The x86 image cannot run natively on ARM64. For JetPack 6.0, start from the
matching NVIDIA L4T TensorRT image, install ROS Humble for ARM64, use the
JetPack-compatible PyTorch wheel, and rebuild the camera engine on that device.
The source algorithms are portable, but this repository's x86 build and measured
performance do not constitute a tested Jetson deployment.

Sources: [NVIDIA TensorRT container release notes](https://docs.nvidia.com/deeplearning/frameworks/container-release-notes/),
[ROS Humble Ubuntu installation](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html),
[Ultralytics package metadata](https://pypi.org/project/ultralytics/8.4.95/).
