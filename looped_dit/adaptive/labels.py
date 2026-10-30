"""Training labels of the controller: per-exit errors of a frozen model and the exit targets."""

from __future__ import annotations

import torch

from ..model import LoopedMMDiT


def exit_targets(losses: torch.Tensor) -> torch.Tensor:
    """Targets y_1..y_{N-1} [..., N-1] from the per-exit errors l_1..l_N [..., N]:

        y_r = max(0, max_{k > r} (l_r - l_k) / (k - r)),

    the largest error reduction per extra loop that the passes after r can still buy."""
    n = losses.shape[-1]
    targets = []
    for r in range(n - 1):
        extra = torch.arange(1, n - r, device=losses.device, dtype=losses.dtype)
        gain = (losses[..., r : r + 1] - losses[..., r + 1 :]) / extra
        targets.append(gain.amax(dim=-1).clamp_min(0))
    return torch.stack(targets, dim=-1)


@torch.no_grad()
def exit_rollout(
    model: LoopedMMDiT,
    x0: torch.Tensor,
    x_t: torch.Tensor,
    text: torch.Tensor,
    text_mask: torch.Tensor,
    cfg_scale: float = 6.0,
    num_loops: int | None = None,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Run the frozen model once with every loop exit decoded.

    Returns the conditional loop states [h_0, ..., h_N] (each an (image, text) pair) and
    the errors l_1..l_N [B, N], l_k = mean((x0_hat(k) - x0)^2), of the guided x0
    predictions x0_hat(k) decoded after k loops against the clean image x0."""
    n = model.num_loops if num_loops is None else num_loops
    b = x_t.shape[0]
    guided = cfg_scale != 1.0
    if guided:  # rows: [conditional; unconditional]
        img, txt = model.embed(torch.cat([x_t, x_t]), torch.cat([text, text]),
                               torch.cat([text_mask, torch.zeros_like(text_mask)]))
    else:
        img, txt = model.embed(x_t, text, text_mask)
    states = [(img[:b], txt[:b])]
    losses = []
    for r in range(1, n + 1):
        img, txt = model.loop_pass(img, txt, r)
        states.append((img[:b], txt[:b]))
        pred = model.decode(img, txt)
        if guided:
            pred = pred[b:] + (pred[:b] - pred[b:]) * cfg_scale
        losses.append((pred - x0.float()).pow(2).mean(dim=(1, 2, 3)))
    return states, torch.stack(losses, dim=1)
