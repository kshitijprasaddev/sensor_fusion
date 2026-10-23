"""Spatial fusion contracts that do not need CUDA, ROS or pretrained models."""

import math
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from group8_fusion.mid_model import BEVMiddleFusion, decode


def sample(height=5, width=7, batch=1):
    torch.manual_seed(23)
    return [torch.randn(batch, channels, height, width) for channels in (192, 128, 8)]


@pytest.mark.parametrize("height,width", [(1, 1), (5, 7), (9, 17)])
def test_arbitrary_spatial_sizes_and_explicit_all_absent_output(height, width):
    model = BEVMiddleFusion().eval()
    inputs = sample(height, width)
    result = model(*inputs, torch.zeros(1, 3, height, width))
    assert result["heatmap_logits"].shape == (1, 10, height, width)
    assert result["box_regression"].shape == (1, 8, height, width)
    assert torch.equal(result["heatmap_logits"], torch.full_like(result["heatmap_logits"], -20))
    assert not result["box_regression"].any()
    assert not result["gates"].any()
    assert all(torch.isfinite(v).all() for v in result.values())


@pytest.mark.parametrize("sensor", [0, 1, 2])
def test_each_single_modality_operates_with_one_normalized_gate(sensor):
    inputs = sample()
    available = torch.zeros(1, 3, 5, 7)
    available[:, sensor] = 1
    result = BEVMiddleFusion().eval()(*inputs, available)
    assert torch.equal(result["gates"], available)
    assert all(torch.isfinite(v).all() for v in result.values())


def test_absent_nan_inf_and_negative_inputs_cannot_contaminate_present_sensor():
    model = BEVMiddleFusion().eval()
    inputs = sample()
    available = torch.zeros(1, 3, 5, 7)
    available[:, 1] = 1
    expected = model(*inputs, available)
    changed = [inputs[0].clone(), inputs[1].clone(), inputs[2].clone()]
    changed[0].fill_(float("nan"))
    changed[2].fill_(-1e30)
    changed[2][:, :2] = float("inf")
    actual = model(*changed, available)
    for name in expected:
        assert torch.equal(expected[name], actual[name])
    # Mixed empty/present windows exercise the attention padding mask.
    partial = torch.zeros(1, 3, 5, 7)
    partial[:, 1, :2, :3] = 1
    out = model(*changed, partial)
    assert all(torch.isfinite(v).all() for v in out.values())
    assert torch.equal(out["heatmap_logits"][:, :, 2:], torch.full_like(out["heatmap_logits"][:, :, 2:], -20))


def test_box_prediction_has_nonzero_gradients_to_all_sensor_features_and_encoders():
    model = BEVMiddleFusion().train()
    inputs = [x.requires_grad_() for x in sample()]
    result = model(*inputs, torch.ones(1, 3, 5, 7))
    # Box loss alone must train every modality, not just the classification path.
    target = torch.linspace(-.5, .5, 8).reshape(1, 8, 1, 1)
    loss = (result["box_regression"] - target).square().mean()
    loss.backward()
    for x in inputs:
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for encoder in (model.camera_encoder, model.lidar_encoder, model.radar_encoder):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
        assert encoder[0].weight.grad.abs().sum() > 0
    assert model.gate.weight.grad.abs().sum() > 0
    assert model.window_attention.attention.in_proj_weight.grad.abs().sum() > 0


def test_absent_sensor_receives_zero_gradient_and_invalid_availability_is_safe():
    model = BEVMiddleFusion().eval()
    inputs = [x.requires_grad_() for x in sample()]
    available = torch.ones(1, 3, 5, 7)
    available[:, 0] = -1
    available[:, 2] = float("nan")
    result = model(*inputs, available)
    result["box_regression"].square().mean().backward()
    assert not inputs[0].grad.any() and not inputs[2].grad.any()
    assert inputs[1].grad.abs().sum() > 0
    assert all(torch.isfinite(v).all() for v in result.values())


def test_batched_mixed_windows_have_zero_or_normalized_gates():
    model = BEVMiddleFusion().eval()
    inputs = sample(9, 17, batch=2)
    available = torch.zeros(2, 3, 9, 17)
    available[1, :, :8, :9] = torch.tensor([.2, .5, 1.]).reshape(3, 1, 1)
    result = model(*inputs, available)
    expected = (available > 0).any(1).float()
    assert torch.allclose(result["gates"].sum(1), expected)
    assert all(torch.isfinite(v).all() for v in result.values())
    assert not result["gates"][0].any()


def test_decode_predicts_known_cell_dimensions_and_yaw_without_proposal_boxes():
    heatmap = torch.full((1, 10, 4, 6), -20.)
    heatmap[0, 3, 1, 4] = 4
    reg = torch.zeros(1, 8, 4, 6)
    reg[0, :, 1, 4] = torch.tensor([
        0., math.log(3), 1.2, math.log(4), math.log(2), math.log(1.5), 1., 0.
    ])
    grid = dict(x_min=0., y_min=-24., cell_size=.75, width=6, height=4)
    result = decode(dict(heatmap_logits=heatmap, box_regression=reg), grid)
    expected = np.array([[3.375, -22.6875, 1.2, 4., 2., 1.5, math.pi / 2, 0., 0.]])
    np.testing.assert_allclose(result["boxes"], expected, atol=1e-6)
    assert result["labels"].tolist() == [3]
    np.testing.assert_allclose(result["scores"], [torch.sigmoid(torch.tensor(4.)).item()])


def test_decode_local_peaks_topk_invalid_values_and_empty_predictions():
    heatmap = torch.full((1, 2, 3, 4), -20.)
    heatmap[0, 0, 0, 0] = 5
    heatmap[0, 0, 0, 1] = 4  # Neighbour is suppressed by the stronger peak.
    heatmap[0, 1, 2, 3] = 3
    reg = torch.zeros(1, 8, 3, 4)
    reg[:, 7] = 1
    grid = dict(x_min=0., y_min=-24., cell_size=.75, width=4, height=3)
    pred = dict(heatmap_logits=heatmap, box_regression=reg)
    assert len(decode(pred, grid)["boxes"]) == 2
    assert decode(pred, grid, max_detections=1)["labels"].tolist() == [0]
    assert decode(pred, grid, max_detections=0)["boxes"].shape == (0, 9)
    reg[0, 2, 0, 0] = float("nan")
    assert decode(pred, grid)["labels"].tolist() == [1]
    assert decode(pred, grid, max_detections=1)["labels"].tolist() == [1]
    heatmap[0, 0, 0, 0] = float("inf")
    assert decode(pred, grid)["labels"].tolist() == [1]
    heatmap.fill_(float("nan"))
    assert not len(decode(pred, grid)["boxes"])


def test_shape_and_grid_mismatches_fail_with_useful_errors():
    model = BEVMiddleFusion().eval()
    inputs = sample()
    with pytest.raises(ValueError, match="Availability"):
        model(*inputs, torch.ones(1, 2, 5, 7))
    result = model(*inputs, torch.ones(1, 3, 5, 7))
    with pytest.raises(ValueError, match="grid dimensions"):
        decode(result, dict(x_min=0, y_min=0, cell_size=.75, width=8, height=5))
