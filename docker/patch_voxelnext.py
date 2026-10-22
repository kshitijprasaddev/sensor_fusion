"""Disable eager Argoverse dataset registration for inference-only deployment.

The project uses DatasetTemplate and recorded ROS clouds, not the AV2 SDK.
No model layers, preprocessing values or learned parameters are changed.
"""

from pathlib import Path
import sysconfig
import shutil

# Upstream setup.py misses namespace-only focal_sparse_conv helpers. Preserve
# compiled wheel extensions while copying the original Python helper files.
shutil.copytree(
    "/opt/voxelnext/pcdet",
    Path(sysconfig.get_paths()["purelib"]) / "pcdet",
    dirs_exist_ok=True,
)
path = Path(sysconfig.get_paths()["purelib"]) / "pcdet/datasets/__init__.py"
s = path.read_text()
s = s.replace("from .argo2.argo2_dataset import Argo2Dataset\n", "")
s = s.replace("    'Argo2Dataset': Argo2Dataset,\n", "")
path.write_text(s)
