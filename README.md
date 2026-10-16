# Sensor Fusion

## PC development (x86_64)

Use `nvcr.io/nvidia/tensorrt:23.08-py3`: Ubuntu 22.04, CUDA 12.2.1, TensorRT 8.6.1.6 and cuDNN 8.9.4. This is a close development match for JetPack 6.0, which uses TensorRT 8.6.2.

Install Docker, an NVIDIA driver supporting CUDA 12.2.1 (535 or newer), and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Complete its Docker configuration steps.

Clone the repo, then start the container:

```bash
git clone https://github.com/kshitijprasaddev/sensor_fusion.git
cd sensor_fusion
docker pull nvcr.io/nvidia/tensorrt:23.08-py3
docker run --rm -it --gpus all \
  -v "$PWD:/workspace/sensor_fusion" \
  -w /workspace/sensor_fusion \
  nvcr.io/nvidia/tensorrt:23.08-py3 bash
```

Inside the container, check GPU access and TensorRT:

```bash
nvidia-smi
python3 -c "import tensorrt as trt; print(trt.__version__)"
```

Work in `/workspace/sensor_fusion`. Files saved here stay on your PC. Packages installed inside the container are removed when it exits.

## Jetson Orin later (ARM64)

Use JetPack 6.0 on the Jetson and `nvcr.io/nvidia/l4t-tensorrt:r8.6.2-devel`.

From the repo folder on Jetson:

```bash
docker pull nvcr.io/nvidia/l4t-tensorrt:r8.6.2-devel
docker run --rm -it --runtime nvidia \
  -v "$PWD:/workspace/sensor_fusion" \
  -w /workspace/sensor_fusion \
  nvcr.io/nvidia/l4t-tensorrt:r8.6.2-devel bash
```

Transfer source code and ONNX models. Build TensorRT `.engine` files on Jetson and rebuild custom C++/CUDA plugins for ARM64. PC engines cannot be reused on Jetson. Test accuracy and performance on the device.

Sources: [PC container versions](https://docs.nvidia.com/deeplearning/frameworks/container-release-notes/), [JetPack 6.0](https://docs.nvidia.com/jetson/jetpack/6.0/release-notes/index.html), [Jetson container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-tensorrt).

