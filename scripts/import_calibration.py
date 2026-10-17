#!/usr/bin/env python3
"""Import ROS camera_calibration YAML into a fresh fusion configuration.

Use the rectified projection matrix for image_rectified. Never silently mark
an approximate or raw-image matrix as verified calibration.
"""
import argparse
from pathlib import Path
import numpy as np
import yaml


def import_calibration(source, base, output):
    with open(source, encoding='utf-8') as stream:
        calibration = yaml.safe_load(stream)
    with open(base, encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    width, height = int(calibration['image_width']), int(calibration['image_height'])
    P = np.asarray(calibration['projection_matrix']['data'], dtype=float).reshape(3, 4)
    K = P[:, :3]
    if width <= 0 or height <= 0 or not np.isfinite(P).all():
        raise ValueError('Invalid image dimensions or projection matrix')
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError('Expected a valid rectified pinhole projection matrix')
    if not np.allclose(P[:, 3], 0):
        raise ValueError('Nonzero projection translation: review stereo frame conventions before import')
    config['camera'].update(width=width, height=height, K=K.ravel().tolist(),
                            status='calibrated', allow_approximate=False,
                            calibration_source=Path(source).name)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}; choose a fresh output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('camera_yaml')
    p.add_argument('--base', default='src/group8_fusion/config/fusion.yaml')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    import_calibration(args.camera_yaml, args.base, args.output)
    print(f'Created {args.output}. Verify extrinsics and reprojection on independent targets before interpreting accuracy.')
