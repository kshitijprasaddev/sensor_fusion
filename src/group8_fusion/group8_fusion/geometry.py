"""Frame transforms, projection and PointCloud2 decoding without ROS imports."""
import numpy as np
from scipy.spatial.transform import Rotation


def transform_matrix(translation, quaternion):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def transform_points(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def project(points, camera_from_lidar, intrinsics, width, height):
    camera = transform_points(points, camera_from_lidar)
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0.1)
    pixels = np.full((len(camera), 2), np.nan)
    p = camera[valid] @ np.asarray(intrinsics).T
    pixels[valid] = p[:, :2] / p[:, 2:3]
    valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
    valid &= (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    return pixels, valid, camera[:, 2]


def box_corners(center, size):
    signs = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
    return center + signs * np.asarray(size) / 2


def iou(a, b):
    lo = np.maximum(a[:2], b[:2])
    hi = np.minimum(a[2:], b[2:])
    intersection = np.prod(np.maximum(0, hi - lo))
    union = np.prod(np.maximum(0, a[2:] - a[:2])) + np.prod(np.maximum(0, b[2:] - b[:2])) - intersection
    return float(intersection / union) if union > 0 else 0.0


def cloud_array(msg, fields=('x', 'y', 'z')):
    """Honor field offsets, row padding, endianness and non-float radar fields.

    Returns all rows including NaNs so callers can keep velocity aligned with XYZ.
    """
    types = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
    by_name = {f.name: f for f in msg.fields}
    endian = '>' if msg.is_bigendian else '<'
    dtype = np.dtype({
        'names': list(fields),
        'formats': [endian + types[by_name[n].datatype] for n in fields],
        'offsets': [by_name[n].offset for n in fields], 'itemsize': msg.point_step})
    data = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                      strides=(msg.row_step, msg.point_step))
    return np.column_stack([data[n].ravel() for n in fields]).astype(np.float64)


def image_array(msg):
    if msg.encoding not in ('bgr8', 'rgb8', 'mono8'):
        raise ValueError(f'Unsupported image encoding: {msg.encoding}')
    channels = 1 if msg.encoding == 'mono8' else 3
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    img = rows[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    if channels == 1:
        img = np.repeat(img, 3, axis=2)
    elif msg.encoding == 'rgb8':
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)
