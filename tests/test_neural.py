"""CPU tests of learned fusion contracts, failure handling and equivariance."""

from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from group8_fusion.attention_fusion import QueryFusion, pack_features, weak_targets
from group8_fusion.processor import load_config


def tensors(m=2, n=4, k=5):
    torch.manual_seed(8)
    return [
        torch.randn(m, 266),
        torch.randn(n, 152),
        torch.randn(k, 8),
        torch.randn(m, n, 5),
    ]


def test_missing_sensors_have_explicit_finite_null_attention():
    model = QueryFusion().eval()
    fused, l, r = model(*tensors(n=0, k=0))
    assert fused.shape == (2, 64) and torch.isfinite(fused).all()
    assert torch.equal(l, torch.ones(2, 1)) and torch.equal(r, torch.ones(2, 1))


def test_candidate_order_does_not_change_fused_evidence():
    model = QueryFusion().eval()
    q, l, r, g = tensors()
    fused, la, ra = model(q, l, r, g)
    li = torch.tensor([3, 0, 2, 1])
    ri = torch.tensor([4, 1, 0, 2, 3])
    f2, l2, r2 = model(q, l[li], r[ri], g[:, li])
    assert torch.allclose(fused, f2, atol=1e-5)
    assert torch.allclose(la[:, :-1][:, li], l2[:, :-1], atol=1e-5)
    assert torch.allclose(ra[:, :-1][:, ri], r2[:, :-1], atol=1e-5)


def test_learning_has_gradient_paths_to_all_three_sensor_features():
    model = QueryFusion()
    args = [v.requires_grad_() for v in tensors()]
    fused, l, r = model(*args)
    loss = -l[:, 0].log().mean() - r[:, 0].log().mean() + fused.square().mean()
    loss.backward()
    assert args[0].grad[:, :256].abs().sum() > 0
    assert args[1].grad[:, :128].abs().sum() > 0
    assert args[2].grad.abs().sum() > 0
    assert torch.isfinite(torch.cat([v.grad.flatten() for v in args])).all()


def test_neural_motion_uses_history_and_has_finite_gradients():
    model = QueryFusion()
    f = torch.randn(2, 64)
    pos = torch.randn(2, 3)
    v, h = model.predict_motion(
        f, pos, pos - 0.1, torch.tensor([0.1, 0.1]), torch.zeros(2, 64)
    )
    v2, _ = model.predict_motion(f, pos, pos - 0.1, torch.tensor([0.1, 0.1]), h)
    assert v.shape == (2, 2) and not torch.allclose(v, v2)
    v.square().mean().backward()
    assert model.motion.weight_hh.grad is not None


def test_empty_clouds_and_invalid_radar_attributes_are_safe():
    track = {
        "id": 1,
        "label": "car",
        "pointer": np.zeros(256),
        "bbox": np.array([40, 40, 60, 60]),
    }
    lidar = {
        "boxes": np.empty((0, 9)),
        "labels": np.empty(0, int),
        "scores": np.empty(0),
        "features": np.empty((0, 128)),
        "names": [],
    }
    raw = np.array([[10, 0, 0, np.inf, np.nan], [np.nan, 0, 0, 0, 0]], float)
    rec = pack_features([track], lidar, raw, np.eye(4), np.eye(4), np.eye(3), 100, 100)
    assert rec["query"].shape == (1, 266) and rec["lidar"].shape == (0, 152)
    assert rec["radar"].shape == (1, 8) and np.isfinite(rec["radar"]).all()
    assert abs(rec["radar"][0, 3]) <= 3
    l, r = weak_targets(rec, [track])
    assert l.tolist() == [0] and r.tolist() == [1]


def test_missing_learned_weights_fail_before_loading_pretrained_models(tmp_path):
    from group8_fusion.neural import NeuralProcessor

    cfg = load_config(
        Path(__file__).parents[1] / "src/group8_fusion/config/fusion_ai.yaml"
    )
    cfg["neural"]["fusion"]["weights"] = str(tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError, match="Train"):
        NeuralProcessor(cfg, np.eye(4), np.eye(4))


def test_v2_radar_supervision_keeps_multiple_plausible_returns_and_null():
    from group8_fusion.robust_fusion import radar_distribution

    boxes = torch.tensor([[10., 0, 0, 4, 2, 2, 0, 0, 0]])
    radar = torch.tensor([[9., 0, 0], [11., 0, 0], [30., 0, 0]])
    targets = radar_distribution(boxes, radar, torch.tensor([0, 1]))
    assert torch.allclose(targets[0], torch.tensor([.5, .5, 0, 0]))
    assert torch.equal(targets[1], torch.tensor([0., 0, 0, 1]))
    empty = radar_distribution(boxes, radar[:0], torch.tensor([0]))
    assert torch.equal(empty, torch.ones(1, 1))


def test_v2_contextual_radar_is_permutation_equivariant_and_handles_empty_sets():
    from group8_fusion.robust_fusion import RobustQueryFusion

    model = RobustQueryFusion().eval()
    q, l, r, g = tensors()
    f, la, ra = model(q, l, r, g)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    f2, la2, ra2 = model(q, l, r[permutation], g)
    assert torch.allclose(f, f2, atol=1e-5)
    assert torch.allclose(la, la2, atol=1e-5)
    assert torch.allclose(ra[:, :-1][:, permutation], ra2[:, :-1], atol=1e-5)
    f, la, ra = model(q, l[:0], r[:0], g[:, :0])
    assert torch.isfinite(f).all()
    assert torch.equal(la, torch.ones(2, 1))
    assert torch.equal(ra, torch.ones(2, 1))


def test_v2_gated_camera_and_contextual_radar_receive_training_gradients():
    from group8_fusion.robust_fusion import RobustQueryFusion

    model = RobustQueryFusion()
    args = [v.requires_grad_() for v in tensors()]
    f, la, ra = model(*args)
    (f.square().mean() - la[:, 0].log().mean() - ra[:, 0].log().mean()).backward()
    assert model.camera_gate.weight.grad.abs().sum() > 0
    assert model.radar_context.self_attn.in_proj_weight.grad.abs().sum() > 0
    assert args[0].grad[:, :256].abs().sum() > 0
    assert args[1].grad[:, :128].abs().sum() > 0
    assert args[2].grad.abs().sum() > 0


def test_checkpoint_versioning_roundtrips_and_rejects_unknown_schema():
    from group8_fusion.robust_fusion import RobustQueryFusion, load_fusion_model

    for model, version in ((QueryFusion(), "v1"), (RobustQueryFusion(), "v2")):
        checkpoint = dict(schema="group8-query-fusion-" + version,
                          supervision="weak_geometry", state_dict=model.state_dict())
        assert type(load_fusion_model(checkpoint)) is type(model)
    with pytest.raises(ValueError, match="schema"):
        load_fusion_model(dict(schema="unknown", supervision="weak_geometry"))


def test_temperature_softens_confidence_without_changing_candidate_order():
    from group8_fusion.robust_fusion import RobustQueryFusion

    p = torch.tensor([[.9, .08, .02]])
    softened = RobustQueryFusion.temper(p, 3.)
    assert softened.argmax(1).item() == p.argmax(1).item()
    assert softened.max() < p.max() and torch.allclose(softened.sum(1), torch.ones(1))
