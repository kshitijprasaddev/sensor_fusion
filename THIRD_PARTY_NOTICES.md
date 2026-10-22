# Third-party components

The project source is provided under AGPL-3.0. Ultralytics code and its downloaded
YOLO checkpoints are subject to Ultralytics licensing, including AGPL-3.0 or a
separate enterprise license. See [Ultralytics licensing](https://www.ultralytics.com/license).
Checkpoint provenance and hashes are recorded by the preparation script.

The NVIDIA base container is governed by NVIDIA's container and deep learning
license terms, displayed when its original entrypoint starts. The Dockerfile
extends that image; it does not redistribute the image in this Git repository.
CUDA, TensorRT, cuDNN, PyTorch, ROS and other dependencies retain their licenses.

The student recordings and course slides were supplied for this project.
They are not included in the repository or relicensed by the project source
license. Generated validation summaries describe local experiments. Use the
recordings and derived imagery according to the permissions granted by their
owner and the course.

The neural profile additionally uses the authors' VoxelNeXt source and nuScenes
checkpoint: [VoxelNeXt](https://github.com/JIA-Lab-research/VoxelNeXt), Apache-2.0
source license. Dataset/checkpoint terms remain applicable. The pinned upstream
source is obtained during the Docker build and is not copied into this repository.

SAM 2.1 is described by [Meta's SAM 2 project](https://github.com/facebookresearch/sam2).
This deployment executes the Ultralytics 8.4.95 adaptation and corresponding
published checkpoint; its distribution follows Ultralytics licensing. It does
not claim to execute the separate official Meta Python package. Model source
URLs, sizes and SHA-256 hashes are captured by `scripts/prepare_ai.py`.

`models/query_fusion.pt` contains this project's locally trained small fusion/GRU
head. It is distributed with the AGPL-3.0 project source, with training provenance
inside the checkpoint. Upstream perception checkpoints, raw recordings and
cached feature arrays are excluded from Git. The literature review distinguishes
reused implementations from ideas, alternatives and unimplemented architectures.
