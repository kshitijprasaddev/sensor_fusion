#!/usr/bin/env python3
"""Emit NUL-separated Docker GPU arguments, without changing host configuration.

Prefer NVIDIA Container Toolkit. The direct mode is an x86 Linux fallback for
this workstation, exposing only NVIDIA device nodes and read-only driver files.
OpenGL uses Mesa software rendering in that mode; CUDA inference still uses GPU.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

mode = os.environ.get('GPU_MODE', 'auto')
if mode not in ('auto', 'toolkit', 'cdi', 'direct'):
    raise SystemExit('GPU_MODE must be auto, toolkit, cdi or direct')
if mode == 'auto':
    info = json.loads(subprocess.check_output(['docker', 'info', '--format', '{{json .Runtimes}}']))
    has_cdi = any(Path(p).exists() for p in ('/var/run/cdi/nvidia.yaml', '/etc/cdi/nvidia.yaml'))
    mode = 'cdi' if has_cdi else ('toolkit' if 'nvidia' in info else 'direct')
args = []
if mode == 'cdi':
    args = ['--device=nvidia.com/gpu=all', '-e', '__NV_PRIME_RENDER_OFFLOAD=1', '-e', '__GLX_VENDOR_LIBRARY_NAME=nvidia']
elif mode == 'toolkit':
    args = ['--gpus', 'all', '-e', '__NV_PRIME_RENDER_OFFLOAD=1', '-e', '__GLX_VENDOR_LIBRARY_NAME=nvidia']
else:
    for name in ('nvidia0', 'nvidiactl', 'nvidia-uvm'):
        path = Path('/dev') / name
        if not path.exists():
            raise SystemExit(f'Missing {path}. Install NVIDIA Container Toolkit or use GPU=0.')
        args += ['--device', str(path)]
    libraries = subprocess.check_output(['ldconfig', '-p'], text=True).splitlines()
    for name in ('libcuda.so.1', 'libnvidia-ml.so.1', 'libnvidia-ptxjitcompiler.so.1', 'libnvidia-nvvm.so.4'):
        matches = [line.split('=>')[-1].strip() for line in libraries if line.split()[0] == name]
        if matches:
            args += ['-v', f'{Path(matches[0]).resolve()}:/usr/lib/x86_64-linux-gnu/{name}:ro']
        elif name == 'libcuda.so.1':
            raise SystemExit('NVIDIA libcuda.so.1 not found')
    if Path('/usr/bin/nvidia-smi').is_file():
        args += ['-v', '/usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro']
    args += ['-e', 'LIBGL_ALWAYS_SOFTWARE=1']
    print('Using direct NVIDIA device access. RViz uses Mesa; CUDA uses the GPU. '
          'Install NVIDIA Container Toolkit for accelerated OpenGL.', file=sys.stderr)
sys.stdout.buffer.write(b'\0'.join(a.encode() for a in args) + b'\0')
