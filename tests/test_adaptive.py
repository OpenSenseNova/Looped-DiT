"""CPU tests of adaptive looping: python -m pytest tests"""

import itertools
import math
from pathlib import Path

import pytest
import torch

from looped_dit.adaptive import (
    LoopController,
    adaptive_euler_sample,
    adaptive_prediction,
    exit_rollout,
    exit_targets,
    resolve_threshold,
)
from looped_dit.adaptive.calibrate import calibrated_only
from looped_dit.adaptive.controller import drop_fixed_settings
from looped_dit.adaptive.data import HoldOut
from looped_dit.adaptive.eval import loop_summary
from looped_dit.adaptive.train import ControllerConfig, policy_exits, policy_metrics, stack_rows, threshold_for_budget
from looped_dit.diffusion import euler_sample

from test_looped_dit import inputs, tiny_model

CFG = 3.0
PARAMETERS = 1_050_053  # with the default sizes (B/32 hidden size 768, 4 loops)


def tiny_controller(model, trained: bool = True) -> LoopController:
    torch.manual_seed(2)
    controller = LoopController(hidden_size=64, num_loops=model.num_loops, token_dim=32, num_heads=4,
                                loop_embed_dim=8, head_dims=(32, 16)).eval()
    if trained:  # the zero-initialised last layer of the timestep MLP has moved
        torch.nn.init.normal_(controller.time_mlp[-1].weight, std=1.0)
    return controller


def guided(model, x, text, mask, loops):
    cond = model(x, text, mask, num_loops=loops)
    uncond = model(x, text, torch.zeros_like(mask), num_loops=loops)
    return uncond + (cond - uncond) * CFG


def test_controller_size():
    assert sum(p.numel() for p in LoopController().parameters()) == PARAMETERS


def test_exit_targets():
    torch.manual_seed(0)
    losses = torch.rand(5, 4)
    expected = torch.zeros(5, 3)
    for i, r in itertools.product(range(5), range(3)):
        expected[i, r] = max(0.0, max((losses[i, r] - losses[i, k]) / (k - r) for k in range(r + 1, 4)))
    assert torch.allclose(exit_targets(losses), expected)
    # No later exit improves: the target is zero.
    assert exit_targets(torch.tensor([[1.0, 2.0, 3.0, 4.0]])).abs().sum() == 0


def test_rollout_matches_model_exits():
    model = tiny_model().eval()
    x0, text, mask = inputs()
    x_t = 0.5 * x0 + torch.randn_like(x0)
    states, losses = exit_rollout(model, x0, x_t, text, mask, CFG)
    assert len(states) == model.num_loops + 1 and losses.shape == (2, model.num_loops)
    for r in range(1, model.num_loops + 1):
        expected = (guided(model, x_t, text, mask, r) - x0).pow(2).mean(dim=(1, 2, 3))
        assert torch.allclose(losses[:, r - 1], expected, atol=1e-5)


@pytest.mark.parametrize("threshold,depth", [(-math.inf, 3), (math.inf, 1)])
def test_extreme_thresholds_give_fixed_depths(threshold, depth):
    model = tiny_model().eval()
    x, text, mask = inputs()
    pred, loops = adaptive_prediction(model, tiny_controller(model), x, text, mask, threshold, 0.3, CFG)
    assert (loops == depth).all()
    assert torch.allclose(pred, guided(model, x, text, mask, depth), atol=1e-5)


def test_mixed_exits_match_per_sample_depths():
    model = tiny_model().eval()
    controller = tiny_controller(model)
    torch.manual_seed(3)
    x, text = torch.randn(6, 3, 32, 32), torch.randn(6, 6, 16)
    mask = torch.ones(6, 6, dtype=torch.long)
    mask[::2, 3:] = 0
    t = torch.linspace(0.0, 0.9, 6)
    # A threshold between the first-pass scores, so the batch splits.
    img, txt = model.embed(x, text, mask)
    first = controller(*model.loop_pass(img, txt, 1), mask, 1, img, txt, t)
    threshold = first.median().item()
    pred, loops = adaptive_prediction(model, controller, x, text, mask, threshold, t, CFG)
    assert (loops == 1).any() and (loops > 1).any()
    for i in range(6):
        expected = guided(model, x[i : i + 1], text[i : i + 1], mask[i : i + 1], int(loops[i]))
        assert torch.allclose(pred[i : i + 1], expected, atol=1e-5)


def test_adaptive_sampler_without_exits_equals_euler():
    model = tiny_model().eval()
    controller = tiny_controller(model)
    _, text, mask = inputs()
    torch.manual_seed(5)
    reference = euler_sample(model, text, mask, 32, steps=4, cfg_scale=CFG)
    torch.manual_seed(5)
    images, loops = adaptive_euler_sample(model, controller, text, mask, 32, -math.inf, steps=4, cfg_scale=CFG)
    assert loops.shape == (2, 4) and (loops == 3).all()
    assert torch.allclose(images, reference, atol=1e-5)


def test_controller_reads_the_timestep():
    model = tiny_model().eval()
    x0, text, mask = inputs()
    states, _ = exit_rollout(model, x0, x0, text, mask, CFG)
    # At initialisation the timestep has no effect: training starts from a timestep-free controller.
    fresh = tiny_controller(model, trained=False)
    assert torch.allclose(fresh(*states[1], mask, 1, *states[0], 0.1), fresh(*states[1], mask, 1, *states[0], 0.9))
    controller = tiny_controller(model)
    t = torch.tensor([0.1, 0.7])
    a = controller(*states[1], mask, 1, *states[0], t)
    assert not torch.allclose(a, controller(*states[1], mask, 1, *states[0], t.flip(0)))
    assert torch.allclose(controller(*states[1], mask, 1, *states[0], 0.1)[0], a[0])  # a number = every sample
    # The sampler passes t: its exits match calling the controller with t directly.
    threshold = a.min().item()
    _, loops = adaptive_prediction(model, controller, x0, text, mask, threshold, t, CFG)
    assert loops.tolist() == [1 if s <= threshold else loops[i].item() for i, s in enumerate(a.tolist())]


def test_controller_rows_and_policy():
    model = tiny_model().eval()
    controller = tiny_controller(model)
    x0, text, mask = inputs()
    states, _ = exit_rollout(model, x0, x0, text, mask, CFG)
    t = torch.tensor([0.2, 0.6])
    scores = controller(*stack_rows(states, mask, t)).view(-1, 2).t()  # [images, exits 1..N-1]
    for r in (1, 2):  # row order: all images at exit 1, then exit 2
        assert torch.allclose(scores[:, r - 1], controller(*states[r], mask, r, *states[r - 1], t))
    scores = torch.tensor([[0.5, 0.2], [0.05, 0.9], [0.3, 0.01]])
    assert policy_exits(scores, 0.1).tolist() == [3, 1, 2]
    assert policy_exits(scores, -1.0).tolist() == [3, 3, 3]
    assert policy_exits(scores, 1.0).tolist() == [1, 1, 1]
    torch.manual_seed(0)
    scores = torch.rand(1000, 3)
    for budget in (1.25, 1.5, 2.0, 3.0):
        assert abs(policy_exits(scores, threshold_for_budget(scores, budget)).float().mean() - budget) < 0.01
    metrics = policy_metrics(scores, torch.rand(1000, 4), [1.5])
    assert set(metrics) == {"rel_err@1.5", "threshold@1.5", "mean_loops@1.5", "rel_err_mean"}


def test_held_out_split():
    captions = [f"caption number {i}" for i in range(4000)]
    held, train = HoldOut(0.05), HoldOut(0.05, held_out=False)
    flags = [held(c) for c in captions]
    assert flags == [held(c) for c in captions] and flags == [not train(c) for c in captions]
    assert 0.03 < sum(flags) / len(flags) < 0.07
    assert held(" caption number 7 ") == held("caption number 7")
    torch.manual_seed(0)
    ids = [torch.randint(0, 1000, (12,)) for _ in range(2000)]
    assert 0.03 < sum(map(held, ids)) / len(ids) < 0.07


def test_threshold_lookup():
    ckpt = {"thresholds": {"1.5": {"threshold": 0.01}, "2": {"threshold": 0.002}}}
    assert resolve_threshold(ckpt, target_loops=2.0) == 0.002
    assert resolve_threshold(ckpt, threshold=0.3) == 0.3
    with pytest.raises(KeyError):
        resolve_threshold(ckpt, target_loops=1.25)


def test_calibration_drops_validation_thresholds():
    thresholds = {"1.5": {"threshold": 0.01, "mean_loops": 1.5, "source": "512 held-out captions"},
                  "3": {"threshold": -0.001, "source": "validation set"}}
    assert set(calibrated_only(thresholds)) == {"1.5"}


def test_controller_config():
    path = Path(__file__).resolve().parents[1] / "configs" / "adaptive" / "b32_controller.yml"
    cfg = ControllerConfig.from_yaml(path)
    assert cfg.batch_size % cfg.micro_batch_size == 0
    assert sum(p.numel() for p in LoopController(**cfg.controller_kwargs(768, 4)).parameters()) == PARAMETERS


def test_development_checkpoint_settings():
    # Checkpoints of the development code record settings that are fixed here.
    kwargs = LoopController().kwargs
    assert drop_fixed_settings({**kwargs, "timestep_input": True, "target": "max_gain"}) == kwargs
    with pytest.raises(ValueError):
        drop_fixed_settings({**kwargs, "timestep_input": False})
    cfg = ControllerConfig.from_dict({"backbone": "b.pt", "target": "max_gain", "label": "x0", "final_steps": 10})
    assert cfg.backbone == "b.pt"


def test_loop_summary(tmp_path):
    assert loop_summary(tmp_path) is None
    (tmp_path / ".loops").mkdir()
    (tmp_path / ".loops" / "000000.json").write_text('{"loops": ["1122", "4444"]}')
    (tmp_path / ".loops" / "000001.json").write_text('{"loops": ["1111"]}')
    assert loop_summary(tmp_path) == {"images": 3, "mean_loops": (1.5 + 4 + 1) / 3}
