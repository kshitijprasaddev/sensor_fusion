"""Teacher target, loss and evaluation safeguards independent of recorded data."""

import math
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from group8_fusion.mid_model import decode
from group8_fusion.mid_training import (
    agreement_metrics, center_targets, chronological_paths, detection_loss,
    shift_camera, teacher_targets,
)


GRID = dict(x_min=0., y_min=-3., cell_size=.75, width=8, height=8)


def boxes(*centers):
    return np.array([[x, y, 0., 4., 2., 1.5, 0., 0., 0.] for x, y in centers],
                    dtype=np.float32).reshape(-1, 9)


def test_teacher_filter_preserves_lower_bound_and_excludes_upper_bound_and_invalid_boxes():
    candidates = boxes((0., -3.), (6., 0.), (3., 3.), (2., 0.), (2., 0.), (2., 0.), (2., 0.))
    candidates[3, 3] = -1
    candidates[4, 0] = np.nan
    scores = np.array([.35, .9, .9, .9, .9, .34, .9], np.float32)
    labels = np.array([0, 0, 0, 0, 0, 0, 10])
    selected, selected_labels, confidence = teacher_targets(candidates, labels, scores, GRID)
    np.testing.assert_array_equal(selected, candidates[:1])
    assert selected_labels.tolist() == [0]
    np.testing.assert_allclose(confidence, [.35])
    with pytest.raises(ValueError, match="equal lengths"):
        teacher_targets(candidates, labels[:1], scores, GRID)


def test_continuous_targets_round_trip_through_decoder_with_correct_yaw_and_dimensions():
    teacher = boxes((2.4, -.6))
    teacher[0, 2] = .7
    teacher[0, 6] = -.8
    target = center_targets(teacher, np.array([4]), np.array([.9]), GRID)
    logits = torch.full((1, 10, 8, 8), -20.)
    logits[0, 4][target["heatmap"][4] == 1] = 5.
    regression = target["regression"].clone()
    occupied = target["mask"].bool()
    # Decode receives raw offset logits, while targets store fractional offsets.
    regression[:2, occupied] = torch.logit(regression[:2, occupied])
    prediction = decode(dict(heatmap_logits=logits, box_regression=regression[None]), GRID)
    np.testing.assert_allclose(prediction["boxes"], teacher, atol=1e-6)
    assert prediction["labels"].tolist() == [4]
    assert target["heatmap"][4].max() == 1


def test_same_cell_collision_supervises_only_the_more_confident_teacher():
    teacher = boxes((2.4, -.6), (2.5, -.5))
    teacher[1, 3] = 5.
    target = center_targets(teacher, np.array([0, 1]), np.array([.6, .95]), GRID)
    assert target["collisions"] == 1
    assert target["mask"].sum() == 1
    assert not target["heatmap"][0].any()
    assert (target["heatmap"][1] == 1).sum() == 1
    np.testing.assert_allclose(target["regression"][3][target["mask"].bool()], [math.log(5.)])


def test_empty_teacher_loss_is_finite_and_suppresses_background_without_box_gradients():
    target = center_targets(boxes(), np.array([], int), np.array([]), GRID)
    prediction = dict(heatmap_logits=torch.zeros(1, 10, 8, 8, requires_grad=True),
                      box_regression=torch.zeros(1, 8, 8, 8, requires_grad=True))
    batched = {key: target[key][None] for key in ("heatmap", "regression", "mask", "weight")}
    loss, components = detection_loss(prediction, batched)
    assert torch.isfinite(loss) and components["box"] == 0 and components["yaw_norm"] == 0
    loss.backward()
    assert (prediction["heatmap_logits"].grad > 0).all()
    assert not prediction["box_regression"].grad.any()


def test_positive_loss_has_finite_box_gradients_only_at_supervised_center():
    teacher = boxes((2.4, -.6))
    target = center_targets(teacher, np.array([0]), np.array([.9]), GRID)
    prediction = dict(heatmap_logits=torch.full((1, 10, 8, 8), -1000., requires_grad=True),
                      box_regression=torch.zeros(1, 8, 8, 8, requires_grad=True))
    batched = {key: target[key][None] for key in ("heatmap", "regression", "mask", "weight")}
    loss, _ = detection_loss(prediction, batched)
    loss.backward()
    for value in prediction.values():
        assert torch.isfinite(value.grad).all()
    active = prediction["box_regression"].grad.abs().sum(1)[0] > 0
    assert torch.equal(active, target["mask"].bool())
    assert prediction["heatmap_logits"].grad[0, 0][target["heatmap"][0] == 1] < 0


@pytest.mark.parametrize("dx,dy", [(1, 1), (-1, -1), (2, -2)])
def test_camera_jitter_uses_zero_padding_and_never_wraps_evidence(dx, dy):
    camera = torch.ones(2, 4, 5)
    availability = torch.ones(4, 5)
    shifted, mask = shift_camera(camera, availability, dx, dy)
    assert shifted.sum() == 2 * (4 - abs(dy)) * (5 - abs(dx))
    assert mask.sum() == (4 - abs(dy)) * (5 - abs(dx))
    assert torch.equal(shifted[0], mask)
    assert camera.sum() == 40 and availability.sum() == 20


def test_teacher_matching_counts_duplicates_and_class_errors_and_wraps_yaw():
    teacher = boxes((2., 0.), (5., 0.))
    teacher[0, 6] = math.pi - .05
    predictions = boxes((2.1, 0.), (2.2, 0.), (5., 0.), (5., 0.))
    predictions[0, 6] = -math.pi + .05
    result = agreement_metrics(
        [dict(boxes=predictions, scores=np.array([.9, .8, .7, .1]), labels=np.array([0, 0, 1, 0]))],
        [dict(boxes=teacher, labels=np.array([0, 0]))], threshold=.2,
    )
    assert result["predictions"] == 3 and result["teacher_objects"] == 2
    assert result["matched_teacher_objects"] == 1
    assert result["teacher_center_precision"] == pytest.approx(1 / 3)
    assert result["teacher_center_recall"] == .5
    assert result["matched_yaw_mean_error_rad"] == pytest.approx(.1, abs=1e-6)
    with pytest.raises(ValueError, match="frame counts"):
        agreement_metrics([], [dict(boxes=teacher, labels=np.array([0, 0]))], .2)


def test_empty_teacher_and_prediction_metrics_have_explicit_zero_counts():
    result = agreement_metrics(
        [dict(boxes=boxes(), scores=np.array([]), labels=np.array([], int))],
        [dict(boxes=boxes(), labels=np.array([], int))], threshold=.2,
    )
    assert result["predictions"] == result["teacher_objects"] == 0
    assert result["teacher_center_f1"] == 0
    assert result["matched_center_xy_mean_error_m"] is None
    assert result["prediction_frame_coverage"] == 0


def test_split_keeps_threshold_calibration_separate_and_purges_adjacent_frames(tmp_path):
    bag1, bag2 = tmp_path / "studentProject", tmp_path / "studentProject1"
    bag1.mkdir()
    bag2.mkdir()
    for index in range(235):
        (bag1 / f"{index:05d}.npz").touch()
    (bag2 / "00000.npz").touch()
    split = chronological_paths(tmp_path)
    frames = {name: {int(path.stem) for path in paths}
              for name, paths in split.items() if name != "development"}
    assert frames["train"] == set(range(164))
    assert frames["selection"] == set(range(188, 211))
    assert frames["calibration"] == set(range(211, 235))
    assert not (set(range(164, 188)) & set.union(*frames.values()))
    assert not (frames["train"] & frames["selection"] or frames["selection"] & frames["calibration"])
    assert all(path.parent == bag2 for path in split["development"])
    (bag1 / "00190.npz").unlink()
    with pytest.raises(ValueError, match="missing.*190"):
        chronological_paths(tmp_path)
