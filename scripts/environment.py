#!/usr/bin/env python3
"""Capture actual installed runtime versions for a validation run."""
import importlib.metadata
import json
import platform
import subprocess
import torch
import tensorrt

data = {'python': platform.python_version(), 'architecture': platform.machine(),
        'tensorrt': tensorrt.__version__, 'torch_cuda_available': torch.cuda.is_available(),
        'torch_cuda_version': torch.version.cuda,
        'torch_cudnn_version': torch.backends.cudnn.version(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'packages': {p: importlib.metadata.version(p) for p in ['torch', 'torchvision',
            'ultralytics', 'numpy', 'scipy', 'opencv-python', 'onnx', 'onnxruntime-gpu']},
        'ros_packages': subprocess.check_output(['dpkg-query', '-W', '-f=${Package} ${Version}\n',
            'ros-humble-rclpy', 'ros-humble-cv-bridge', 'ros-humble-sensor-msgs',
            'ros-humble-tf2-ros', 'ros-humble-rviz2'], text=True).strip().splitlines()}
print(json.dumps(data, indent=2))
