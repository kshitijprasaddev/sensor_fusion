FROM nvcr.io/nvidia/tensorrt:23.08-py3

ENV DEBIAN_FRONTEND=noninteractive ROS_DISTRO=humble \
    PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    QT_X11_NO_MITSHM=1 MPLCONFIGDIR=/tmp/matplotlib
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Keep the CUDA 12.2.1 and TensorRT 8.6.1.6 shipped by the requested image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg locales software-properties-common \
    libgl1 libglib2.0-0 libxcb-xinerama0 mesa-utils xauth \
    python3-pip python3-dev python3-venv && \
    locale-gen en_US.UTF-8 && add-apt-repository universe && \
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main' \
      > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop ros-humble-sensor-msgs-py ros-humble-tf2-ros \
    ros-humble-vision-msgs ros-humble-rosbag2-storage-default-plugins \
    python3-colcon-common-extensions python3-pytest && \
    rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# The supplied NVIDIA nv24.05 PyTorch build targets Jetson. Use the corresponding
# upstream x86 CUDA wheel here instead of mixing ARM and x86 packages.
RUN python3 -m pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 \
      --index-url https://download.pytorch.org/whl/cu121
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
ENV OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1

WORKDIR /workspace/sensor_fusion
COPY . .
RUN source /opt/ros/humble/setup.bash && colcon build --symlink-install \
    && python3 -m pytest -q tests
ENTRYPOINT ["/workspace/sensor_fusion/docker/entrypoint.sh"]
CMD ["bash"]
