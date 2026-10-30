"""The loop controller: after each loop pass it scores whether another pass is worth running.

After pass r < N the controller reads the conditional loop state h_r (image and text
tokens) and the timestep t, and predicts y_r, the largest reduction of the x0 error
that further loops can still buy, per extra loop (see labels.exit_targets). Sampling
continues while the score is above a threshold and exits at the first pass where it
is not; pass N always exits. The threshold sets the compute budget, so one
controller serves every average loop count (calibrate.py finds the threshold of a
budget).

Architecture: a single cross-attention layer. Four learned queries, shifted by an
embedding of the loop index plus an embedding of t, attend to the projected image
and text tokens; an MLP head maps the mean query, the loop/time embedding and the
RMS change of the image and text states over the last pass to the score
(1.05M parameters with the defaults).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from ..model import RMSNorm


class LoopController(nn.Module):
    """Scores a loop state: higher means more to gain from another loop."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_loops: int = 4,
        token_dim: int = 256,
        num_queries: int = 4,
        num_heads: int = 8,
        ffn_ratio: int = 2,
        loop_embed_dim: int = 64,
        head_dims: tuple[int, int] = (256, 64),
    ):
        super().__init__()
        self.kwargs = dict(hidden_size=hidden_size, num_loops=num_loops, token_dim=token_dim, num_queries=num_queries,
                           num_heads=num_heads, ffn_ratio=ffn_ratio, loop_embed_dim=loop_embed_dim,
                           head_dims=list(head_dims))
        self.num_loops = num_loops
        self.img_norm = RMSNorm(hidden_size)
        self.txt_norm = RMSNorm(hidden_size)
        self.img_proj = nn.Linear(hidden_size, token_dim)
        self.txt_proj = nn.Linear(hidden_size, token_dim)
        self.img_type = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.txt_type = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.txt_valid = nn.Embedding(2, token_dim)  # prompt token vs padding slot
        self.queries = nn.Parameter(torch.randn(num_queries, token_dim) * token_dim**-0.5)
        self.loop_embed = nn.Embedding(num_loops + 1, loop_embed_dim)
        self.loop_to_query = nn.Linear(loop_embed_dim, token_dim)
        # Sinusoidal features of t -> MLP, added to the loop embedding. The last layer
        # starts at zero, so training starts from a timestep-free controller.
        self.time_mlp = nn.Sequential(
            nn.Linear(loop_embed_dim, loop_embed_dim), nn.SiLU(), nn.Linear(loop_embed_dim, loop_embed_dim)
        )
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)
        self.query_norm = nn.LayerNorm(token_dim)
        self.token_norm = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, token_dim * ffn_ratio), nn.GELU(), nn.Linear(token_dim * ffn_ratio, token_dim)
        )
        width = token_dim + loop_embed_dim + 2
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, head_dims[0]),
            nn.GELU(),
            nn.Linear(head_dims[0], head_dims[1]),
            nn.GELU(),
            nn.Linear(head_dims[1], 1),
        )

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        text_mask: torch.Tensor,
        loop: int | torch.Tensor,
        prev_img: torch.Tensor,
        prev_txt: torch.Tensor,
        t: float | torch.Tensor,
    ) -> torch.Tensor:
        """img/txt: the state after pass `loop` [B, tokens, hidden]; prev_img/prev_txt: the
        state before it; text_mask [B, L]; t: the timestep in [0, 1] (t = 1 is the clean
        image), a number or [B]. Returns float32 scores [B].

        Runs in float32 whatever the autocast state (thresholds can be small)."""
        with torch.autocast(img.device.type, enabled=False):
            img, txt, prev_img, prev_txt = (z.float() for z in (img, txt, prev_img, prev_txt))
            valid = text_mask.to(device=img.device, dtype=torch.bool)
            b = img.shape[0]
            if isinstance(loop, int):
                loop = torch.full((b,), loop, device=img.device, dtype=torch.long)
            # RMS change over the last pass; for text, over the prompt tokens only.
            d_img = (img - prev_img).pow(2).mean(dim=(1, 2)).sqrt()
            w = valid.float()[:, :, None]
            d_txt = ((txt - prev_txt).pow(2) * w).sum(dim=(1, 2)) / (w.sum(dim=(1, 2)) * txt.shape[-1]).clamp_min(1)
            d_txt = d_txt.sqrt()

            tokens = torch.cat(
                [
                    self.txt_proj(self.txt_norm(txt)) + self.txt_type + self.txt_valid(valid.long()),
                    self.img_proj(self.img_norm(img)) + self.img_type,
                ],
                dim=1,
            )
            ignore = torch.cat([~valid, torch.zeros(b, img.shape[1], device=img.device, dtype=torch.bool)], dim=1)
            t = torch.as_tensor(t, device=img.device, dtype=torch.float32).expand(b)
            loop_emb = self.loop_embed(loop)
            loop_emb = loop_emb + self.time_mlp(timestep_features(t, loop_emb.shape[-1]))
            query = self.queries.unsqueeze(0) + self.loop_to_query(loop_emb).unsqueeze(1)
            keys = self.token_norm(tokens)
            query = query + self.attn(self.query_norm(query), keys, keys, key_padding_mask=ignore, need_weights=False)[0]
            query = query + self.ffn(self.ffn_norm(query))
            features = torch.cat([query.mean(dim=1), loop_emb, d_img[:, None], d_txt[:, None]], dim=-1)
            return self.head(features).squeeze(-1)


def timestep_features(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal features [B, dim] of timesteps t in [0, 1] [B]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    angles = 1000.0 * t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


# Settings that development checkpoints record and this controller has fixed.
FIXED_SETTINGS = {"timestep_input": True, "target": "max_gain", "label": "x0"}


def drop_fixed_settings(values: dict) -> dict:
    """`values` without the keys of FIXED_SETTINGS (and the unused final-image label options),
    after checking that they hold the fixed value."""
    values = dict(values)
    for key, fixed in FIXED_SETTINGS.items():
        value = values.pop(key, fixed)
        if value != fixed:
            raise ValueError(f"checkpoint has {key}={value!r}; this controller requires {fixed!r}")
    for key in ("final_steps", "final_pool"):
        values.pop(key, None)
    return values


def load_controller(path: str | Path, device: torch.device) -> tuple[LoopController, dict]:
    """A controller checkpoint (written by adaptive.train / adaptive.calibrate): returns the
    controller (float32, eval mode) and the checkpoint dict, whose "thresholds" map a target
    average loop count to its calibrated threshold."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    controller = LoopController(**drop_fixed_settings(ckpt["controller_kwargs"]))
    controller.load_state_dict(ckpt["controller"])
    return controller.to(device).eval().requires_grad_(False), ckpt


def resolve_threshold(ckpt: dict, target_loops: float | None = None, threshold: float | None = None) -> float:
    """The threshold to sample with: given directly, or the calibrated one of a target."""
    if (target_loops is None) == (threshold is None):
        raise ValueError("give exactly one of target_loops and threshold")
    if threshold is not None:
        return float(threshold)
    calibrated = ckpt.get("thresholds") or {}
    for key, value in calibrated.items():
        if math.isclose(float(key), float(target_loops), abs_tol=1e-6):
            return float(value["threshold"])
    raise KeyError(f"no threshold calibrated for {target_loops} loops (available: {sorted(calibrated, key=float)}); "
                   "run looped_dit.adaptive.calibrate first")
