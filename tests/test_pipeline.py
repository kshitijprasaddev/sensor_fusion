from types import SimpleNamespace as NS
import numpy as np
import pytest
from group8_fusion.geometry import transform_matrix, transform_points, project, cloud_array, image_array, iou
from group8_fusion.perception import Object3D, dbscan, remove_ground, associate_camera, associate_radar
from group8_fusion.tracking import Tracker
from group8_fusion.processor import load_config
from pathlib import Path


def obj(x, y=0., label='car'):
    return Object3D(np.array([x, y, 1.], dtype=float), np.ones(3), label)


def test_extrinsic_direction_and_inverse():
    T = transform_matrix([1, 2, 3], [0, 0, np.sqrt(0.5), np.sqrt(0.5)])
    p = np.array([[1., 0., 0.]])
    assert np.allclose(transform_points(p, T), [[1, 3, 3]])
    assert np.allclose(transform_points(transform_points(p, T), np.linalg.inv(T)), p)


def test_projection_rejects_behind_and_outside():
    K = [[100, 0, 50], [0, 100, 50], [0, 0, 1]]
    uv, valid, _ = project(np.array([[0., 0, 2], [0, 0, -2], [10, 0, 1]]), np.eye(4), K, 100, 100)
    assert valid.tolist() == [True, False, False]
    assert np.allclose(uv[0], [50, 50])


@pytest.mark.parametrize('big', [False, True])
def test_cloud_honors_padding_and_endian(big):
    endian = '>' if big else '<'
    buffer = bytearray(80)
    dtype = np.dtype({'names': ['x', 'y', 'z', 'RCS'], 'formats': [endian+'f4']*3+['i1'],
                      'offsets': [0, 4, 8, 12], 'itemsize': 16})
    data = np.ndarray((2, 2), dtype, buffer=buffer, strides=(40, 16))
    data['x'] = [[1, 2], [3, 4]]
    data['y'] = 5
    data['z'] = 6
    data['RCS'] = -8
    fields = [NS(name=n, offset=o, datatype=t) for n, o, t in [('x',0,7),('y',4,7),('z',8,7),('RCS',12,1)]]
    msg = NS(fields=fields, data=buffer, height=2, width=2, point_step=16, row_step=40, is_bigendian=big)
    actual = cloud_array(msg, ('x', 'y', 'z', 'RCS'))
    assert np.allclose(actual[:, 0], [1, 2, 3, 4])
    assert np.all(actual[:, 3] == -8)


def test_rgb_image_with_row_padding():
    msg = NS(encoding='rgb8', data=bytes([1,2,3,4,5,6,99,99]), height=1, width=2, step=8)
    assert image_array(msg).tolist() == [[[3,2,1],[6,5,4]]]


def test_dbscan_noise_does_not_form_object():
    p = np.array([[0.,0], [.1,0], [0,.1], [4,4], [4.1,4], [4,4.1], [10,10]])
    groups = dbscan(p, .3, 3)
    assert [g.tolist() for g in groups] == [[0,1,2], [3,4,5]]
    assert dbscan(np.empty((0, 3)), .3, 3) == []


def test_ground_removal_preserves_vertical_object():
    ground = np.array([[x,y,-1.5] for x in range(10) for y in range(10)], dtype=float)
    elevated = np.array([[2.,2.,0.], [2.,2.,1.]])
    out = remove_ground(np.vstack([ground, elevated]))
    assert np.allclose(out, elevated)


def test_camera_only_labels_projected_matches():
    objects = [Object3D(np.array([0.,0.,10.]), np.ones(3))]
    boxes = [{'xyxy': [40,40,60,60], 'label': 'car', 'confidence': .9}]
    assert associate_camera(objects, boxes, np.eye(4), [[100,0,50],[0,100,50],[0,0,1]], 100,100) == 1
    assert objects[0].label == 'car'
    assert objects[0].sources == ['lidar','camera']


def test_radar_gating_and_variance_weighting():
    lidar = obj(10.)
    radar = obj(11.)
    radar.sources = ['radar']
    radar.variance = np.array([1.,1.])
    far = obj(60.)
    far.sources = ['radar']
    objects = [lidar]
    assert associate_radar(objects, [radar, far], 2.5) == 1
    assert 10 < lidar.center[0] < 10.5
    assert np.all(lidar.variance < .16)
    assert len(objects) == 2


def test_kalman_tracks_motion_and_keeps_ids():
    tracker = Tracker()
    for i in range(20):
        tracks = tracker.update([obj(10 + i * .1)], i * .1)
    assert len(tracks) == 1 and tracks[0].id == 1
    assert abs(tracks[0].state[2] - 1) < .15
    assert np.linalg.eigvalsh(tracks[0].covariance).min() > 0


def test_tracker_misses_expiry_and_clock_rewind():
    tracker = Tracker()
    tracker.update([obj(10)], 10.)
    assert tracker.update([obj(10)], 10.1)[0].id == 1
    assert not tracker.update([], 11.)
    assert not tracker.update([obj(10)], 1.)
    assert tracker.update([obj(10)], 1.1)[0].id == 2


def test_tracker_distant_object_never_steals_id():
    tracker = Tracker(min_hits=1)
    tracker.update([obj(10)], 0.)
    tracks = tracker.update([obj(30)], .1)
    assert {t.id for t in tracks} == {1,2}


def test_iou_empty_and_overlap():
    assert iou(np.array([0,0,10,10]), np.array([0,0,10,10])) == 1
    assert iou(np.array([0,0,0,0]), np.array([0,0,0,0])) == 0


def test_default_config_marks_approximate_calibration():
    cfg = load_config(Path(__file__).parents[1] / 'src/group8_fusion/config/fusion.yaml')
    assert cfg['camera']['status'] == 'approximate'
    assert cfg['camera']['allow_approximate']
