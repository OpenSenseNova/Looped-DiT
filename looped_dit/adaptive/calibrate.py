"""Calibrate the thresholds of a controller for target average loop counts.

    torchrun --nproc_per_node=8 -m looped_dit.adaptive.calibrate \
        --controller outputs/b32_controller/checkpoints/controller_0020000.pt \
        --target-loops 1.25 1.5 2.0 --out outputs/b32_controller/controller_calibrated.pt

The average loop count is measured where it is spent: on full sampling runs
(100 Euler steps, guidance 6), averaged over images and denoising steps. The prompts
are held-out captions of the controller's training data (the same hash split as its
validation set) or, with --prompt-file, any prompt list. Every run uses the same
prompts and noise, so the average is a deterministic, almost monotone function of
the threshold.

The search runs over the offline budget tau in [1, N]: the threshold whose average
exit is tau on the validation rows stored in the checkpoint. tau = target is the
starting guess (validation rows are corrupted training images, not sampling
trajectories, so it is usually a little off); bracketed secant steps then match the
target to --tolerance. The output is the controller checkpoint with the calibrated
thresholds; pass it to sample / eval with --target-loops.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from ..pipeline import TextEncoder, load_model
from ..utils import atomic_save, init_distributed, is_main, rank, world_size
from .controller import load_controller
from .sampler import adaptive_euler_sample
from .train import ControllerConfig, threshold_for_budget


def held_out_prompts(ckpt: dict, backbone_cfg, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Token ids and masks of `count` distinct held-out captions (same on every rank)."""
    cfg = ControllerConfig.from_dict(ckpt["config"])
    ids, masks = [], []
    if is_main():
        seen = set()
        for batch in cfg.loader(backbone_cfg, held_out=True, batch=16, seed_offset=20_000, workers=1):
            for i, m in zip(batch["input_ids"], batch["attention_mask"]):
                key = tuple(i[m.bool()].tolist())
                if key not in seen:
                    seen.add(key)
                    ids.append(i)
                    masks.append(m)
            if len(ids) >= count:
                break
    out = [torch.stack(ids[:count]) if ids else None, torch.stack(masks[:count]) if masks else None]
    if world_size() > 1:
        dist.broadcast_object_list(out, src=0)
    return out[0], out[1]


def file_prompts(path: str, tokenizer_name: str, length: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    prompts = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()][:count]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=length)
    tokens = tokenizer(prompts, max_length=length, padding="max_length", truncation=True, return_tensors="pt")
    return tokens["input_ids"], tokens["attention_mask"]


class LoopMeter:
    """Average loop count of the calibration set at a threshold (all ranks)."""

    def __init__(self, model, controller, text_encoder, ids, masks, args, image_size, noise_scale, device):
        self.model, self.controller, self.args = model, controller, args
        self.image_size, self.noise_scale, self.device = image_size, noise_scale, device
        dtype = next(model.parameters()).dtype
        mine = list(range(rank(), ids.shape[0], world_size()))
        self.batches = []
        for start in range(0, len(mine), args.batch_size):
            index = mine[start : start + args.batch_size]
            i, m = ids[index].to(device), masks[index].to(device)
            self.batches.append((index[0], text_encoder.encode(i, m).to(dtype), m))
        self.images = ids.shape[0]

    def __call__(self, threshold: float) -> float:
        total = torch.zeros((), device=self.device, dtype=torch.float64)
        for first, text, mask in self.batches:
            torch.manual_seed(self.args.seed + first)  # same noise for every threshold
            _, loops = adaptive_euler_sample(self.model, self.controller, text, mask, self.image_size, threshold,
                                             self.args.steps, self.args.cfg_scale, self.noise_scale)
            total += loops.sum()
        if world_size() > 1:
            dist.all_reduce(total)
        return total.item() / (self.images * self.args.steps)


def calibrated_only(thresholds: dict) -> dict:
    """Drop the thresholds found on the validation set during training: they are not
    calibrated on sampling runs, and a target must not silently fall back to one."""
    return {k: v for k, v in thresholds.items() if v.get("source") != "validation set"}


def calibrate(meter: LoopMeter, val_scores: torch.Tensor, target: float, tolerance: float, max_evals: int):
    """Find tau in [1, N] with meter(threshold(tau)) = target. Returns (best measurement, history)."""
    n = val_scores.shape[1] + 1
    history = []

    def measure(tau: float) -> float:
        threshold = threshold_for_budget(val_scores, tau)
        loops = meter(threshold)
        history.append({"tau": tau, "threshold": threshold, "mean_loops": loops})
        if is_main():
            print(f"  target {target:g}: tau {tau:.4f} threshold {threshold:.6g} -> {loops:.4f} loops", flush=True)
        return loops

    lo, hi = (1.0, 1.0), (float(n), float(n))  # (tau, measured), assumed to bracket the target
    tau = target
    for _ in range(max_evals):
        loops = measure(tau)
        if abs(loops - target) <= tolerance:
            break
        if loops < target:
            lo = (tau, loops)
        else:
            hi = (tau, loops)
        # Secant step inside the bracket; bisect if it lands too close to an end.
        slope = (hi[1] - lo[1]) / (hi[0] - lo[0]) if hi[0] > lo[0] else 0
        guess = lo[0] + (target - lo[1]) / slope if slope > 0 else (lo[0] + hi[0]) / 2
        width = hi[0] - lo[0]
        tau = min(max(guess, lo[0] + 0.05 * width), hi[0] - 0.05 * width)
    best = min(history, key=lambda h: abs(h["mean_loops"] - target))
    return best, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate controller thresholds to target average loop counts.")
    parser.add_argument("--controller", required=True, help="controller checkpoint from looped_dit.adaptive.train")
    parser.add_argument("--target-loops", type=float, nargs="+", required=True)
    parser.add_argument("--out", help="output checkpoint (default: <controller>_calibrated.pt)")
    parser.add_argument("--backbone", help="override the backbone checkpoint path stored in the controller")
    parser.add_argument("--text-encoder", help="override the text encoder path")
    parser.add_argument("--prompt-file", help="one prompt per line (default: held-out training captions)")
    parser.add_argument("--prompts", type=int, default=512, help="number of calibration prompts (one image each)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=0.005, help="in average loops")
    parser.add_argument("--max-evals", type=int, default=10, help="sampling runs per target")
    args = parser.parse_args()

    device = init_distributed()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    controller, ckpt = load_controller(args.controller, device)
    model, backbone_cfg = load_model(args.backbone or ckpt["backbone"], device, weights=ckpt["config"]["weights"])
    text_encoder_name = args.text_encoder or ckpt["config"].get("text_encoder") or backbone_cfg.text_encoder
    if args.prompt_file:
        ids, masks = file_prompts(args.prompt_file, text_encoder_name, backbone_cfg.prompt_length, args.prompts)
    else:
        ids, masks = held_out_prompts(ckpt, backbone_cfg, args.prompts)
    text_encoder = TextEncoder(text_encoder_name, backbone_cfg.prompt_length, device)
    meter = LoopMeter(model, controller, text_encoder, ids, masks, args, backbone_cfg.image_size,
                      backbone_cfg.noise_scale, device)
    del text_encoder
    torch.cuda.empty_cache()

    val_scores = ckpt["val_scores"].float()
    thresholds = calibrated_only(ckpt.get("thresholds") or {})
    report = {}
    for target in args.target_loops:
        start = time.time()
        best, history = calibrate(meter, val_scores, target, args.tolerance, args.max_evals)
        thresholds[f"{target:g}"] = {
            "threshold": best["threshold"],
            "mean_loops": best["mean_loops"],
            "source": f"{ids.shape[0]} {'prompts of ' + args.prompt_file if args.prompt_file else 'held-out captions'}"
                      f", {args.steps} steps, guidance {args.cfg_scale:g}, seed {args.seed}",
        }
        report[f"{target:g}"] = {**best, "history": history, "seconds": time.time() - start}
    if is_main():
        out = Path(args.out or Path(args.controller).with_name(Path(args.controller).stem + "_calibrated.pt"))
        ckpt["thresholds"] = thresholds
        ckpt.pop("optimizer", None)
        atomic_save(ckpt, out)
        out.with_suffix(".json").write_text(json.dumps(report, indent=2))
        for key, value in report.items():
            print(f"target {key}: threshold {value['threshold']:.6g}, {value['mean_loops']:.4f} loops", flush=True)
        print(f"saved {out}", flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
