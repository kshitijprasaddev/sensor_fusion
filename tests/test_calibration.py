from pathlib import Path
import importlib.util
import yaml
import pytest

root = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('import_calibration', root / 'scripts/import_calibration.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_import_uses_rectified_projection_and_keeps_original(tmp_path):
    source, out = tmp_path/'camera.yaml', tmp_path/'calibrated.yaml'
    source.write_text(yaml.safe_dump({'image_width':1920, 'image_height':1200,
        'camera_matrix': {'data':[900,0,800,0,900,500,0,0,1]},
        'projection_matrix': {'data':[1000,0,960,0,0,1000,600,0,0,0,1,0]}}))
    base = root / 'src/group8_fusion/config/fusion.yaml'
    before = base.read_bytes()
    module.import_calibration(source, base, out)
    config = yaml.safe_load(out.read_text())
    assert config['camera']['K'][0] == 1000
    assert config['camera']['status'] == 'calibrated'
    assert config['camera']['allow_approximate'] is False
    assert base.read_bytes() == before
    with pytest.raises(FileExistsError):
        module.import_calibration(source, base, out)


def test_import_rejects_invalid_projection(tmp_path):
    source = tmp_path/'camera.yaml'
    source.write_text(yaml.safe_dump({'image_width':1920, 'image_height':1200,
        'projection_matrix': {'data':[-1,0,960,0,0,1000,600,0,0,0,1,0]}}))
    with pytest.raises(ValueError):
        module.import_calibration(source, root/'src/group8_fusion/config/fusion.yaml', tmp_path/'out.yaml')
