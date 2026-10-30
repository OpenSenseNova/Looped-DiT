"""Looped-DiT denoiser.

The MiniT2I denoiser, a pixel-space variant of MMDiT (patchified image tokens
and T5 text tokens with modality-specific weights and joint attention, no
timestep conditioning), whose double-stream blocks are split into three stages:

    pre-loop   A   blocks[:pre]              run once
    looped     B   the next `core` blocks    run N times
    post-loop  C   the last `post` blocks    run once, followed by the head

    h_0 = A(x),   h_r = B(h_{r-1}),   x0_hat(r) = C(h_r),   r = 1..N

With shared weights (the default) B is one set of blocks reused N times, so the
loop adds depth but no weights. Any loop state h_r can be decoded through C:
deep supervision trains those intermediate exits, and inference can run with a
different loop depth. XSA and the attention gate act on the looped blocks only.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = math.ceil(hidden_dim / 8) * 8
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        for layer in (self.w1, self.w3, self.w2):
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Rotary position embeddings: 1D over text positions, 2D over the patch grid.
# ---------------------------------------------------------------------------

_ROPE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _autocast_state() -> tuple:
    try:
        enabled = torch.is_autocast_enabled("cuda")
        return enabled, torch.get_autocast_dtype("cuda") if enabled else None
    except TypeError:  # torch < 2.4
        enabled = torch.is_autocast_enabled()
        return enabled, torch.get_autocast_gpu_dtype() if enabled else None


def _rope_tables(grid: int | None, n: int, d: int, device, dtype, theta: float = 10000.0):
    """cos/sin tables for 1D (grid=None) or 2D rotary embeddings.

    Tables are built under whatever autocast state the caller runs in (under bf16
    autocast the angle products are computed in bf16, which is what the models
    were trained with), so that state is part of the cache key.
    """
    key = (grid, n, d, str(device), dtype, _autocast_state())
    if key not in _ROPE_CACHE:
        if grid is None:
            inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / d))
            pos = torch.arange(n, device=device, dtype=torch.float32)
            angles = torch.einsum("n,f->nf", pos, inv)
            angles = torch.cat([angles, angles], dim=-1)
        else:
            half = d // 2
            inv = 1.0 / (theta ** (torch.arange(0, half, 2, device=device, dtype=torch.float32) / half))
            freqs = torch.einsum("n,f->nf", torch.arange(grid, device=device, dtype=torch.float32), inv)
            f_h, f_w = torch.broadcast_tensors(freqs[:, None, :], freqs[None, :, :])
            angles = torch.cat([f_h, f_w], dim=-1)
            angles = torch.cat([angles, angles], dim=-1).reshape(n, d)
        _ROPE_CACHE[key] = (angles.cos()[None, None].to(dtype), angles.sin()[None, None].to(dtype))
    return _ROPE_CACHE[key]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, grid: int | None = None) -> torch.Tensor:
    """x: [batch, heads, tokens, head_dim]."""
    cos, sin = _rope_tables(grid, x.shape[2], x.shape[3], x.device, x.dtype)
    return x * cos + rotate_half(x) * sin


def sincos_2d(dim: int, grid: int) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    omega = 1.0 / (10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4)))
    out_y = torch.einsum("n,d->nd", y.flatten().float(), omega)
    out_x = torch.einsum("n,d->nd", x.flatten().float(), omega)
    return torch.cat([out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], dim=1)


def exclusive_self_attention(out: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """XSA (Zhai, 2026: https://arxiv.org/abs/2603.09078):
    remove from each token's attention output the component along that token's
    own value vector, so attention only writes content from other tokens.
    `out` and `v` are token-aligned, heads first."""
    v_hat = F.normalize(v.float(), dim=-1)
    out_f = out.float()
    return (out_f - (out_f * v_hat).sum(dim=-1, keepdim=True) * v_hat).to(out.dtype)


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Two-stage patch embedding: a low-rank patch projection, then a 1x1 conv."""

    def __init__(self, patch_size: int, in_channels: int, hidden_size: int, bottleneck: int):
        super().__init__()
        self.proj1 = nn.Conv2d(in_channels, bottleneck, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(bottleneck, hidden_size, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.proj1.weight)
        nn.init.xavier_uniform_(self.proj2.weight)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)


class TextBlock(nn.Module):
    """Text-only transformer block that refines the T5 tokens before the joint blocks."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, mlp_ratio: float):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, num_heads * head_dim * 3)
        self.proj = nn.Linear(num_heads * head_dim, hidden_size)
        self.mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, txt: torch.Tensor) -> torch.Tensor:
        b, n, _ = txt.shape
        q, k, v = self.qkv(self.norm1(txt)).view(b, n, 3, self.num_heads, self.head_dim).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (self.q_norm(q), self.k_norm(k), v))
        out = F.scaled_dot_product_attention(apply_rope(q), apply_rope(k), v, scale=self.head_dim**-0.5)
        txt = txt + self.proj(out.transpose(1, 2).reshape(b, n, -1))
        return txt + self.mlp(self.norm2(txt))


class DoubleStreamBlock(nn.Module):
    """MMDiT block: separate image/text weights, one joint attention over both.

    `use_xsa` / `use_attn_gate` turn on self-modulating attention (set only for the
    looped blocks). The attention gate (Qiu et al., 2026:
    https://arxiv.org/abs/2505.06708) is head-wise:
    y_i <- y_i * sigmoid(W_g u_i + b_g), with u_i the block's normed input.
    
    `update_text=False` skips the text-stream update, for the last block, whose
    text output is never read.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float,
        grid: int,
        use_xsa: bool = False,
        use_attn_gate: bool = False,
        update_text: bool = True,
    ):
        super().__init__()
        self.num_heads, self.head_dim, self.grid = num_heads, head_dim, grid
        self.use_xsa, self.use_attn_gate, self.update_text = use_xsa, use_attn_gate, update_text
        inner = num_heads * head_dim
        self.img_norm1 = RMSNorm(hidden_size)
        self.img_norm2 = RMSNorm(hidden_size)
        self.txt_norm1 = RMSNorm(hidden_size)
        self.txt_norm2 = RMSNorm(hidden_size)
        self.img_qkv = nn.Linear(hidden_size, inner * 3)
        self.txt_qkv = nn.Linear(hidden_size, inner * 3)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.img_proj = nn.Linear(inner, hidden_size)
        self.txt_proj = nn.Linear(inner, hidden_size)
        if use_attn_gate:
            # Zero bias: the gates start half open on average.
            self.img_gate = nn.Linear(hidden_size, num_heads)
            self.txt_gate = nn.Linear(hidden_size, num_heads)
            nn.init.zeros_(self.img_gate.bias)
            nn.init.zeros_(self.txt_gate.bias)
        self.img_mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        self.txt_mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))

    def forward(self, img: torch.Tensor, txt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, li, _ = img.shape
        lt = txt.shape[1]
        img_n, txt_n = self.img_norm1(img), self.txt_norm1(txt)
        qi, ki, vi = self.img_qkv(img_n).view(b, li, 3, self.num_heads, self.head_dim).unbind(2)
        qt, kt, vt = self.txt_qkv(txt_n).view(b, lt, 3, self.num_heads, self.head_dim).unbind(2)
        # Joint sequence [text; image], heads first.
        q = torch.cat([qt, qi], dim=1).transpose(1, 2)
        k = torch.cat([kt, ki], dim=1).transpose(1, 2)
        v = torch.cat([vt, vi], dim=1).transpose(1, 2)
        q = torch.cat([apply_rope(self.q_norm(q[:, :, :lt])), apply_rope(self.q_norm(q[:, :, lt:]), self.grid)], dim=2)
        k = torch.cat([apply_rope(self.k_norm(k[:, :, :lt])), apply_rope(self.k_norm(k[:, :, lt:]), self.grid)], dim=2)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5)
        if self.use_xsa:
            out = exclusive_self_attention(out, v)
        out = out.transpose(1, 2)  # [b, tokens, heads, head_dim]
        out_t, out_i = out[:, :lt], out[:, lt:]
        if self.use_attn_gate:
            out_i = out_i * torch.sigmoid(self.img_gate(img_n)).unsqueeze(-1)
        img = img + self.img_proj(out_i.reshape(b, li, -1))
        img = img + self.img_mlp(self.img_norm2(img))
        if self.update_text:
            if self.use_attn_gate:
                out_t = out_t * torch.sigmoid(self.txt_gate(txt_n)).unsqueeze(-1)
            txt = txt + self.txt_proj(out_t.reshape(b, lt, -1))
            txt = txt + self.txt_mlp(self.txt_norm2(txt))
        return img, txt


# ---------------------------------------------------------------------------
# Looped MMDiT
# ---------------------------------------------------------------------------


class LoopedMMDiT(nn.Module):
    """Predicts the clean image x0 from a noisy image and T5 text embeddings.

    loop_split = (pre, core, post) partitions the double-stream blocks and
    num_loops = N is the trained loop depth (N = 1 is the MiniT2I model without
    looping). With share_loop_weights=False every pass gets its own copy of the
    core blocks: the compute-matched "deeper" baseline with the same exits.
    """

    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 32,
        in_channels: int = 3,
        hidden_size: int = 768,
        num_heads: int = 12,
        head_dim: int = 64,
        mlp_ratio: float = 2.6667,
        pca_channels: int = 128,
        text_dim: int = 1024,
        text_preamble_depth: int = 2,
        loop_split: tuple[int, int, int] = (6, 5, 6),
        num_loops: int = 4,
        share_loop_weights: bool = True,
        use_xsa: bool = False,
        use_attn_gate: bool = False,
    ):
        super().__init__()
        if len(loop_split) != 3 or min(loop_split) < 1:
            raise ValueError(f"loop_split must be three positive block counts (pre, core, post), got {loop_split}")
        pre, core, post = (int(n) for n in loop_split)
        if num_loops < 1:
            raise ValueError(f"num_loops must be >= 1, got {num_loops}")
        self.patch_size, self.in_channels = patch_size, in_channels
        self.grid = image_size // patch_size
        self.pre, self.core, self.post = pre, core, post
        self.num_loops = int(num_loops)
        self.share_loop_weights = bool(share_loop_weights)

        self.img_embed = PatchEmbed(patch_size, in_channels, hidden_size, pca_channels)
        self.txt_embed = nn.Linear(text_dim, hidden_size, bias=False)
        # Replaces the T5 embedding at padded prompt positions (and everywhere
        # for the unconditional branch of classifier-free guidance).
        self.mask_token = nn.Parameter(torch.zeros(1, 1, text_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        # MiniT2I's timestep and pooled-text embedders. The model has no timestep
        # conditioning and never uses them; they are kept (frozen, see below) so that
        # the model and its checkpoints match MiniT2I and the paper.
        self.t_embed = nn.ModuleDict(
            {"mlp": nn.Sequential(nn.Linear(256, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))}
        )
        for layer in (self.t_embed.mlp[0], self.t_embed.mlp[2]):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
        self.pooled_embed = nn.Linear(text_dim, hidden_size, bias=False)
        self.register_buffer("pos_embed", sincos_2d(hidden_size, self.grid)[None], persistent=False)
        self.txt_blocks = nn.ModuleList(
            TextBlock(hidden_size, num_heads, head_dim, mlp_ratio) for _ in range(text_preamble_depth)
        )
        looped = core if self.share_loop_weights else core * self.num_loops
        depth = pre + looped + post
        self.blocks = nn.ModuleList(
            DoubleStreamBlock(
                hidden_size,
                num_heads,
                head_dim,
                mlp_ratio,
                self.grid,
                use_xsa=use_xsa and pre <= i < pre + looped,
                use_attn_gate=use_attn_gate and pre <= i < pre + looped,
                update_text=i < depth - 1,
            )
            for i in range(depth)
        )
        self.final_norm = RMSNorm(hidden_size)
        self.final = nn.Linear(hidden_size, patch_size * patch_size * in_channels)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)
        # Frozen because they never receive a gradient: the unused embedders and the
        # text-stream update of the last block (whose text output is never read).
        last = self.blocks[-1]
        for module in (self.t_embed, self.pooled_embed, last.txt_norm2, last.txt_proj, last.txt_mlp):
            module.requires_grad_(False)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        p, c, g = self.patch_size, self.in_channels, int(n**0.5)
        x = x.view(b, g, g, p, p, c).permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, c, g * p, g * p)

    def loop_blocks(self, r: int) -> nn.ModuleList:
        """Core blocks run on loop pass r (1-based)."""
        start = self.pre if self.share_loop_weights else self.pre + (r - 1) * self.core
        return self.blocks[start : start + self.core]

    def embed(self, x: torch.Tensor, text: torch.Tensor, text_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Embeddings and pre-loop blocks: inputs -> loop state h_0 (image and text tokens)."""
        text = torch.where(text_mask.to(torch.bool)[:, :, None], text, self.mask_token.to(text.dtype))
        img = self.img_embed(x) + self.pos_embed.to(device=x.device, dtype=x.dtype)
        txt = self.txt_embed(text)
        for block in self.txt_blocks:
            txt = block(txt)
        for block in self.blocks[: self.pre]:
            img, txt = block(img, txt)
        return img, txt

    def loop_pass(self, img: torch.Tensor, txt: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Loop pass r (1-based): h_{r-1} -> h_r."""
        for block in self.loop_blocks(r):
            img, txt = block(img, txt)
        return img, txt

    def decode(self, img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
        """Post-loop blocks and output head: a loop state -> x0 prediction."""
        for block in self.blocks[len(self.blocks) - self.post :]:
            img, txt = block(img, txt)
        return self.unpatchify(self.final(self.final_norm(img))).float()

    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        num_loops: int | None = None,
        exit_loops: tuple[int, ...] = (),
    ) -> torch.Tensor | tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """x: noisy images [B, C, H, W]; text: T5 states [B, L, text_dim];
        text_mask: [B, L], 1 for prompt tokens (all 0 = unconditional).

        num_loops overrides the loop depth at inference. exit_loops lists
        intermediate depths r < num_loops to decode as well; the call then
        returns (final prediction, {r: prediction after r loops}).
        """
        n = self.num_loops if num_loops is None else int(num_loops)
        if n < 1 or (not self.share_loop_weights and n > self.num_loops):
            raise ValueError(f"num_loops={n} is not available for this model (trained with {self.num_loops})")
        exits = sorted({int(r) for r in exit_loops})
        if any(not 1 <= r < n for r in exits):
            raise ValueError(f"exit_loops must lie in [1, {n}), got {exits}")

        img, txt = self.embed(x, text, text_mask)
        states = {}
        for r in range(1, n + 1):
            img, txt = self.loop_pass(img, txt, r)
            if r in exits:
                states[r] = (img, txt)
        out = self.decode(img, txt)
        if not exits:
            return out
        return out, {r: self.decode(*states[r]) for r in exits}
