"""Sampling with adaptive loop depth: every sample exits on its own at every denoising step.

The unconditional branch of guidance exits at the same depth as the conditional one.
"""

from __future__ import annotations

import torch
from PIL import Image

from ..model import LoopedMMDiT
from ..pipeline import TextEncoder, to_pil
from .controller import LoopController


@torch.no_grad()
def adaptive_prediction(
    model: LoopedMMDiT,
    controller: LoopController,
    x: torch.Tensor,
    text: torch.Tensor,
    text_mask: torch.Tensor,
    threshold: float,
    t: float | torch.Tensor,
    cfg_scale: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Guided x0 prediction at timestep t (a number or [B]) where every sample exits on its own.

    Returns the prediction [B, C, H, W] and the number of loops each sample ran [B].
    Samples that exit drop out of the batch, so later passes only cost the samples
    still looping."""
    n = controller.num_loops
    if n > model.num_loops and not model.share_loop_weights:
        raise ValueError(f"the controller runs {n} loops, the model has {model.num_loops}")
    b = x.shape[0]
    t = torch.as_tensor(t, device=x.device, dtype=torch.float32).expand(b)
    guided = cfg_scale != 1.0
    if guided:  # rows: [conditional; unconditional]
        img, txt = model.embed(torch.cat([x, x]), torch.cat([text, text]),
                               torch.cat([text_mask, torch.zeros_like(text_mask)]))
    else:
        img, txt = model.embed(x, text, text_mask)
    out_img, out_txt = torch.empty_like(img), torch.empty_like(txt)
    loops = torch.full((b,), n, device=x.device, dtype=torch.long)
    active = torch.arange(b, device=x.device)  # samples still looping
    rows = torch.cat([active, active + b]) if guided else active  # their rows in img/txt
    for r in range(1, n + 1):
        prev_img, prev_txt = img, txt
        img, txt = model.loop_pass(img, txt, r)
        k = active.numel()
        if r == n:
            stop = torch.ones(k, device=x.device, dtype=torch.bool)
        else:
            score = controller(img[:k], txt[:k], text_mask[active], r, prev_img[:k], prev_txt[:k], t[active])
            stop = score <= threshold
            if not stop.any():
                continue
        loops[active[stop]] = r
        stop_rows = torch.cat([stop, stop]) if guided else stop
        out_img[rows[stop_rows]] = img[stop_rows]
        out_txt[rows[stop_rows]] = txt[stop_rows]
        keep = ~stop_rows
        active, rows, img, txt = active[~stop], rows[keep], img[keep], txt[keep]
        if active.numel() == 0:
            break
    pred = model.decode(out_img, out_txt)
    if guided:
        pred = pred[b:] + (pred[:b] - pred[b:]) * cfg_scale
    return pred, loops


@torch.no_grad()
def adaptive_euler_sample(
    model: LoopedMMDiT,
    controller: LoopController,
    text: torch.Tensor,
    text_mask: torch.Tensor,
    image_size: int,
    threshold: float,
    steps: int = 100,
    cfg_scale: float = 6.0,
    noise_scale: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """diffusion.euler_sample with adaptive loop depth.

    Returns the images and the loop counts [B, steps] of every denoising step."""
    was_training = model.training
    model.eval()
    controller.eval()
    b = text.shape[0]
    device = text.device
    x = torch.randn(b, 3, image_size, image_size, device=device, dtype=text.dtype) * noise_scale
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    amp_dtype = text.dtype if text.dtype in (torch.float16, torch.bfloat16) else torch.float32
    loops = torch.zeros(b, steps, device=device, dtype=torch.long)
    for i in range(steps):
        t0, t1 = ts[i], ts[i + 1]
        with torch.autocast("cuda", dtype=amp_dtype, enabled=x.is_cuda and amp_dtype != torch.float32):
            pred, loops[:, i] = adaptive_prediction(model, controller, x, text, text_mask, threshold, t0, cfg_scale)
        v = (pred - x) / (1.0 - t0).clamp_min(0.05)
        x = x + v * (t1 - t0)
    model.train(was_training)
    return x.clamp(-1, 1), loops


@torch.no_grad()
def generate_adaptive(
    model: LoopedMMDiT,
    controller: LoopController,
    text_encoder: TextEncoder,
    prompts: list[str],
    threshold: float,
    image_size: int = 512,
    steps: int = 100,
    cfg_scale: float = 6.0,
    noise_scale: float = 2.0,
) -> tuple[list[Image.Image], torch.Tensor]:
    """pipeline.generate with adaptive loop depth: returns the images and the loop count of
    every image and denoising step [len(prompts), steps]. Seed torch beforehand."""
    input_ids, mask = text_encoder.tokenize(prompts)
    dtype = next(model.parameters()).dtype
    text = text_encoder.encode(input_ids, mask).to(dtype)
    images, loops = adaptive_euler_sample(model, controller, text, mask, image_size, threshold, steps, cfg_scale,
                                          noise_scale)
    return to_pil(images), loops.cpu()
