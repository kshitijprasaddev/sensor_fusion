"""Validity and geometry checks, not claims of detection accuracy."""

from types import SimpleNamespace
import numpy as np
from group8_fusion.quality import observation_issue, instance_points


def test_short_gap_position_jump_is_rejected_without_clipping_the_box():
    box = np.array([6.7, -8.6, 0.0, 4.0, 2.0, 1.5, 0.0])
    original = box.copy()
    previous = (10.0, np.array([6.8, 0.5, 0.0]))
    assert observation_issue(box, 10.1, previous, {}) == "position_jump"
    assert np.array_equal(box, original)
    box[1] = 0.6
    assert observation_issue(box, 10.1, previous, {}) is None
    assert observation_issue(original, 10.1, previous, {"enabled": False}) is None


def test_invalid_geometry_and_timestamp_do_not_pass_confidence_gates():
    box = np.array([10.0, 0, 0, 4, 2, 1.5, 0])
    assert observation_issue(box, float("nan"), None, {}) == "invalid_geometry"
    box[3] = -1
    assert observation_issue(box, 1.0, None, {}) == "invalid_geometry"


def test_instance_points_use_yaw_and_keep_only_existing_points():
    obj = SimpleNamespace(
        center=np.array([10.0, 0, 0]), size=np.array([4.0, 2.0, 2.0]), yaw=np.pi / 2
    )
    track = SimpleNamespace(id=3, object=obj, state=np.array([10.0, 0, 0, 0]))
    points = np.array([[10.0, 1.8, 0], [11.8, 0, 0], [10.0, 0, 2], [np.nan, 0, 0]])
    xyz, identifiers = instance_points(points, [track])
    assert np.allclose(xyz, points[:1]) and identifiers.tolist() == [3]
    assert len(instance_points(points, [])[0]) == 0


def test_overlapping_instances_do_not_duplicate_points():
    obj = SimpleNamespace(
        center=np.array([10.0, 0, 0]), size=np.array([4.0, 2.0, 2.0]), yaw=0.0
    )
    one = SimpleNamespace(id=1, object=obj, state=np.array([10.0, 0, 0, 0]))
    two = SimpleNamespace(id=2, object=obj, state=np.array([11.0, 0, 0, 0]))
    xyz, identifiers = instance_points(
        np.array([[10.1, 0, 0], [10.9, 0, 0]]), [one, two]
    )
    assert len(xyz) == 2 and identifiers.tolist() == [1, 2]


def test_sam_reset_clears_prompt_capacity_counters_with_memory():
    from group8_fusion.neural_models import SAMMemoryTracker

    tracker = SAMMemoryTracker.__new__(SAMMemoryTracker)
    tracker.predictor = SimpleNamespace(memory_bank=[1], obj_idx_set={1})
    tracker.labels, tracker.confidences, tracker.last_masks = (
        {1: "car"},
        {1: 0.9},
        {1: np.ones((2, 2))},
    )
    tracker.next_slot, tracker.capacity_rejections = 8, 3
    tracker.reset()
    assert tracker.capacity_rejections == tracker.next_slot == 0
    assert not tracker.predictor.memory_bank and not tracker.predictor.obj_idx_set
