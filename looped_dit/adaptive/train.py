"""Train the loop controller (controller.py) on a frozen Looped-DiT.

    torchrun --nproc_per_node=8 -m looped_dit.adaptive.train --config configs/adaptive/b32_controller.yml \
        --output-dir outputs/b32_controller

The labels are computed online, so nothing is cached to disk. Each step draws
image-caption pairs from the fine-tuning data, corrupts each image at a timestep of
the sampling grid (t = i / sampling_steps, the inputs the 100-step sampler feeds the
model), runs the frozen model once with every loop exit decoded (conditional and
unconditional, combined with the guidance scale of sampling) and trains the
controller to regress the exit targets y_r of its loop states (labels.exit_targets).
A fresh corruption per visit lets a few epochs of the fine-tuning set give as many
distinct training rows as needed.

A fixed share of the captions (by hash) is held out (data.py). Every `val_every`
steps the controller is scored on a fixed validation set drawn from it: the
regression loss, and, for each budget in `val_budgets`, the mean error l of the exits
it picks at the threshold that gives that average loop count, relative to running a
fixed loop depth with the same average (`val/rel_err@B`, lower is better). Every
checkpoint is kept; calibrate the chosen one on sampling runs with
looped_dit.adaptive.calibrate.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from ..config import TrainConfig, load_yaml
from ..pipeline import TextEncoder, load_model
from ..utils import atomic_save, init_distributed, is_main, rank, seed_everything, world_size
from .controller import LoopController, drop_fixed_settings
from .data import split_loader
from .labels import exit_rollout, exit_targets


@dataclass
class ControllerConfig:
    backbone: str = ""  # Looped-DiT checkpoint (kept frozen)
    weights: str = "ema"
    text_encoder: str | None = None  # default: the backbone's
    cfg_scale: float = 6.0  # guidance scale of sampling
    sampling_steps: int = 100  # timesteps t = i / sampling_steps, i = 0..sampling_steps-1

    # Controller architecture (controller.LoopController).
    token_dim: int = 256
    num_queries: int = 4
    num_heads: int = 8
    ffn_ratio: int = 2
    loop_embed_dim: int = 64
    head_dims: list[int] = field(default_factory=lambda: [256, 64])

    # Data, as in TrainConfig. Captions whose hash falls below holdout_fraction are
    # never trained on; the validation set and the calibration prompts come from them.
    dataset: str = "webdataset"
    chunk_dirs: list[str] = field(default_factory=list)
    webdataset_sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    shuffle_buffer: int = 1000
    num_workers: int = 8
    prefetch_factor: int = 4
    holdout_fraction: float = 0.05
    val_samples: int = 4096

    # Optimization. One image gives num_loops - 1 rows (exits 1..N-1).
    batch_size: int = 256  # images per step
    micro_batch_size: int = 32  # images per GPU and forward pass
    num_steps: int = 20_000
    warmup_steps: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    huber_delta: float = 1.0  # on the targets divided by their validation-set maximum
    seed: int = 0

    log_every: int = 50
    val_every: int = 500
    val_budgets: list[float] = field(default_factory=lambda: [1.25, 1.5, 2.0, 3.0])
    ckpt_every: int = 1000  # every checkpoint is kept (a few MB each)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ControllerConfig:
        cfg = cls()
        names = {f.name for f in fields(cls)}
        for key, value in drop_fixed_settings(values).items():
            if key not in names:
                raise KeyError(f"unknown config key: {key}")
            default = getattr(cfg, key)
            if type(default) in (int, float) and isinstance(value, str):
                value = type(default)(float(value))
            elif type(default) is float and type(value) is int:
                value = float(value)
            setattr(cfg, key, value)
        if not cfg.backbone:
            raise ValueError("set `backbone` to a Looped-DiT checkpoint")
        if not 0 < cfg.holdout_fraction < 1:
            raise ValueError("holdout_fraction must lie in (0, 1)")
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> ControllerConfig:
        return cls.from_dict({**load_yaml(path), **(overrides or {})})

    def controller_kwargs(self, hidden_size: int, num_loops: int) -> dict[str, Any]:
        return dict(hidden_size=hidden_size, num_loops=num_loops, token_dim=self.token_dim,
                    num_queries=self.num_queries, num_heads=self.num_heads, ffn_ratio=self.ffn_ratio,
                    loop_embed_dim=self.loop_embed_dim, head_dims=list(self.head_dims))

    def data_config(self, backbone: TrainConfig, micro_batch_size: int) -> TrainConfig:
        """The backbone's config with this data section, for looped_dit.data."""
        values = backbone.to_dict()
        values.update(dataset=self.dataset, chunk_dirs=self.chunk_dirs, webdataset_sources=self.webdataset_sources,
                      shuffle_buffer=self.shuffle_buffer, num_workers=self.num_workers,
                      prefetch_factor=self.prefetch_factor, micro_batch_size=micro_batch_size, seed=self.seed)
        if self.text_encoder:  # its tokenizer
            values["text_encoder"] = self.text_encoder
        return TrainConfig.from_dict(values)

    def loader(self, backbone: TrainConfig, held_out: bool, batch: int, seed_offset: int = 0,
               workers: int | None = None):
        return split_loader(self.data_config(backbone, batch), self.holdout_fraction, held_out, batch,
                            seed_offset, workers)


def to_unit(pixels: torch.Tensor, device: torch.device) -> torch.Tensor:
    return pixels.to(device, non_blocking=True).float().mul_(1 / 127.5).add_(-1.0)


def corrupt(images: torch.Tensor, step_index: torch.Tensor, steps: int, noise: torch.Tensor) -> torch.Tensor:
    t = (step_index.float() / steps)[:, None, None, None]
    return images * t + noise * (1.0 - t)


@torch.no_grad()
def make_rows(backbone, text_encoder, batch: dict, step_index: torch.Tensor, noise: torch.Tensor,
              cfg: ControllerConfig, device: torch.device):
    """Controller inputs and errors of one batch of images: (states, text mask, errors [B, N])."""
    images = to_unit(batch["pixel_values"], device)
    ids = batch["input_ids"].to(device, non_blocking=True)
    mask = batch["attention_mask"].to(device, non_blocking=True)
    dtype = next(backbone.parameters()).dtype
    with torch.autocast("cuda", dtype=torch.bfloat16):
        text = text_encoder.encode(ids, mask).to(dtype)
        x_t = corrupt(images, step_index.to(device), cfg.sampling_steps, noise.to(device)).to(dtype)
        states, losses = exit_rollout(backbone, images, x_t, text, mask, cfg.cfg_scale)
    return states, mask, losses


def stack_rows(states, mask, t):
    """Rows of exits 1..N-1: (img, txt, mask, loop index, previous img, previous txt, t).
    t: the timestep of each image [B]."""
    n = len(states) - 1
    img = torch.cat([states[r][0] for r in range(1, n)])
    txt = torch.cat([states[r][1] for r in range(1, n)])
    prev_img = torch.cat([states[r - 1][0] for r in range(1, n)])
    prev_txt = torch.cat([states[r - 1][1] for r in range(1, n)])
    b = mask.shape[0]
    loop = torch.arange(1, n, device=mask.device).repeat_interleave(b)
    return img, txt, mask.repeat(n - 1, 1), loop, prev_img, prev_txt, t.to(mask.device).float().repeat(n - 1)


def policy_exits(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    """Exit loop (1..N) of each row of scores [M, N-1] under the threshold rule."""
    stop = torch.cat([scores <= threshold, torch.ones_like(scores[:, :1], dtype=torch.bool)], dim=1)
    return stop.float().argmax(dim=1) + 1  # first exit that stops


def threshold_for_budget(scores: torch.Tensor, budget: float, iters: int = 60) -> float:
    """Threshold whose average exit over the rows of scores [M, N-1] is closest to budget (bisection)."""
    lo, hi = scores.min().item() - 1.0, scores.max().item() + 1.0  # lo: never stop early, hi: always exit at 1
    for _ in range(iters):
        mid = (lo + hi) / 2
        if policy_exits(scores, mid).float().mean().item() > budget:
            lo = mid
        else:
            hi = mid
    return float(min((lo, hi), key=lambda th: abs(policy_exits(scores, th).float().mean().item() - budget)))


def policy_metrics(scores: torch.Tensor, losses: torch.Tensor, budgets: list[float]) -> dict[str, float]:
    """Error of the adaptive exits relative to fixed depths of the same average loop count."""
    fixed = losses.mean(dim=0)  # fixed depth r: mean of l_r
    out = {}
    for budget in budgets:
        threshold = threshold_for_budget(scores, budget)
        exits = policy_exits(scores, threshold)
        mean_exit = exits.float().mean().item()
        chosen = losses.gather(1, (exits - 1)[:, None]).mean().item()
        lo = min(max(int(math.floor(mean_exit)), 1), losses.shape[1])
        hi = min(lo + 1, losses.shape[1])
        frac = mean_exit - lo
        reference = (1 - frac) * fixed[lo - 1].item() + frac * fixed[hi - 1].item()  # mix of two fixed depths
        out[f"rel_err@{budget:g}"] = chosen / reference
        out[f"threshold@{budget:g}"] = threshold
        out[f"mean_loops@{budget:g}"] = mean_exit
    out["rel_err_mean"] = sum(out[f"rel_err@{b:g}"] for b in budgets) / len(budgets)
    return out


def gather(tensor: torch.Tensor) -> torch.Tensor:
    if world_size() == 1:
        return tensor
    parts = [torch.empty_like(tensor) for _ in range(world_size())]
    dist.all_gather(parts, tensor.contiguous())
    return torch.cat(parts)


class ValidationSet:
    """A fixed set of held-out images, timesteps and noise; its loop states stay on the GPU."""

    def __init__(self, cfg: ControllerConfig, backbone, backbone_cfg: TrainConfig, text_encoder, device):
        per_rank = cfg.val_samples // world_size()
        batch = min(cfg.micro_batch_size, per_rank)
        # One loader worker per GPU: with more (rank, worker) pairs than shards, several
        # would read the same shard and the validation set would repeat images.
        loader = cfg.loader(backbone_cfg, held_out=True, batch=batch, seed_offset=10_000, workers=1)
        self.chunks = []
        seen = 0
        for data in loader:
            index = torch.arange(seen, seen + batch) * world_size() + rank()  # global row ids
            step_index = index % cfg.sampling_steps  # balanced over the timestep grid
            g = torch.Generator().manual_seed(cfg.seed * 1_000_003 + int(index[0]))
            noise = torch.randn(batch, 3, backbone_cfg.image_size, backbone_cfg.image_size, generator=g)
            noise *= backbone_cfg.noise_scale
            states, mask, losses = make_rows(backbone, text_encoder, data, step_index, noise, cfg, device)
            t = step_index.float() / cfg.sampling_steps
            self.chunks.append((stack_rows(states, mask, t), losses))
            seen += batch
            if seen >= per_rank:
                break
        del loader
        self.losses = torch.cat([losses for _, losses in self.chunks])
        self.targets = exit_targets(self.losses)

    @torch.no_grad()
    def evaluate(self, controller: LoopController, label_scale: float,
                 cfg: ControllerConfig) -> tuple[dict[str, float], torch.Tensor]:
        """Validation metrics, and the scores [rows, N-1] of all validation images."""
        controller.eval()
        scores, loss, count = [], 0.0, 0
        for rows, losses in self.chunks:
            b = losses.shape[0]
            score = controller(*rows)
            targets = exit_targets(losses).t().reshape(-1) / label_scale  # rows ordered as stack_rows
            loss += F.huber_loss(score, targets, delta=cfg.huber_delta, reduction="sum")
            count += score.numel()
            scores.append(score.view(-1, b).t())  # [B, N-1]
        controller.train()
        scores = torch.cat(scores)
        stats = torch.stack([torch.as_tensor(loss, device=scores.device, dtype=torch.float32).reshape(()),
                             torch.tensor(float(count), device=scores.device)])
        if world_size() > 1:
            dist.all_reduce(stats)
        all_scores, all_losses = gather(scores), gather(self.losses)
        metrics = {"val/loss": (stats[0] / stats[1]).item()}
        metrics.update({f"val/{k}": v for k, v in policy_metrics(all_scores, all_losses, cfg.val_budgets).items()})
        return metrics, all_scores


def lr_at(step: int, cfg: ControllerConfig) -> float:
    return cfg.learning_rate * min(1.0, (step + 1) / max(cfg.warmup_steps, 1))


def checkpoints(directory: Path) -> list[Path]:
    return sorted(directory.glob("controller_*.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the adaptive-looping controller.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    overrides = {k: yaml.safe_load(v) for k, v in (item.split("=", 1) for item in args.set)}
    cfg = ControllerConfig.from_yaml(args.config, overrides)
    device = init_distributed()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out_dir = Path(args.output_dir)
    per_step = cfg.micro_batch_size * world_size()
    if cfg.batch_size % per_step:
        raise ValueError(f"batch_size {cfg.batch_size} is not a multiple of micro_batch_size x GPUs = {per_step}")
    accum = cfg.batch_size // per_step

    backbone, backbone_cfg = load_model(cfg.backbone, device, weights=cfg.weights)
    if backbone_cfg.num_loops < 2:
        raise ValueError("the backbone does not loop")
    text_encoder = TextEncoder(cfg.text_encoder or backbone_cfg.text_encoder, backbone_cfg.prompt_length, device)
    torch.manual_seed(cfg.seed)
    kwargs = cfg.controller_kwargs(backbone_cfg.hidden_size, backbone_cfg.num_loops)
    controller = LoopController(**kwargs).to(device)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    t0 = time.time()
    val = ValidationSet(cfg, backbone, backbone_cfg, text_encoder, device)
    step, label_scale = 0, None
    existing = checkpoints(out_dir / "checkpoints")
    if existing:
        ckpt = torch.load(existing[-1], map_location="cpu", weights_only=True)
        controller.load_state_dict(ckpt["controller"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step, label_scale = int(ckpt["step"]), float(ckpt["label_scale"])
        if is_main():
            print(f"resumed from {existing[-1]}", flush=True)
    if label_scale is None:
        # Targets are divided by their maximum on the validation set, so they are about
        # in [0, 1] and the Huber loss (delta 1) is mostly quadratic.
        label_scale = val.targets.max()
        if world_size() > 1:
            dist.all_reduce(label_scale, op=dist.ReduceOp.MAX)
        label_scale = float(label_scale)
    seed_everything(cfg.seed + 1 + rank() + 7919 * step)
    net = DDP(controller, device_ids=[device.index]) if world_size() > 1 else controller
    if is_main():
        n_params = sum(p.numel() for p in controller.parameters())
        print(f"controller: {n_params:,} parameters; validation: {val.losses.shape[0] * world_size()} images, "
              f"built in {time.time() - t0:.0f}s; label scale {label_scale:.5g}; "
              f"{world_size()} GPUs x {cfg.micro_batch_size} images x {accum} = {cfg.batch_size} images/step", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    wandb = None
    if args.wandb and is_main():
        import wandb

        wandb.init(project="looped-dit-controller", name=out_dir.name, id=out_dir.name, resume="allow",
                   config=asdict(cfg))

    def save(metrics: dict[str, float], val_scores: torch.Tensor) -> None:
        # Thresholds of the validation set, kept for reference; sampling needs calibrated ones.
        thresholds = {f"{b:g}": {"threshold": metrics[f"val/threshold@{b:g}"], "source": "validation set"}
                      for b in cfg.val_budgets}
        state = {"step": step, "controller": controller.state_dict(), "controller_kwargs": kwargs,
                 "optimizer": optimizer.state_dict(), "label_scale": label_scale, "config": asdict(cfg),
                 "backbone": cfg.backbone, "cfg_scale": cfg.cfg_scale, "val_metrics": metrics,
                 "thresholds": thresholds, "val_scores": val_scores.float().cpu()}
        path = out_dir / "checkpoints" / f"controller_{step:07d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_save(state, path)

    batches = iter(cfg.loader(backbone_cfg, held_out=False, batch=cfg.micro_batch_size, seed_offset=1 + step))
    log_file = out_dir / "metrics.jsonl"
    loss_sum, count, last_time, last_step, data_time = 0.0, 0, time.time(), step, 0.0
    net.train()
    while step < cfg.num_steps:
        for micro in range(accum):
            tick = time.time()
            data = next(batches)
            data_time += time.time() - tick
            b = data["input_ids"].shape[0]
            step_index = torch.randint(0, cfg.sampling_steps, (b,))
            noise = torch.randn(b, 3, backbone_cfg.image_size, backbone_cfg.image_size, device=device)
            states, mask, losses = make_rows(backbone, text_encoder, data, step_index,
                                             noise * backbone_cfg.noise_scale, cfg, device)
            no_sync = world_size() > 1 and micro < accum - 1
            with net.no_sync() if no_sync else contextlib.nullcontext():
                rows = stack_rows(states, mask, step_index.float() / cfg.sampling_steps)
                targets = exit_targets(losses).t().reshape(-1) / label_scale  # rows ordered as stack_rows
                loss = F.huber_loss(net(*rows), targets, delta=cfg.huber_delta)
                (loss / accum).backward()
            loss_sum += loss.detach()
            count += 1
        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % cfg.log_every == 0:
            now = time.time()
            log = {"step": step, "loss": float(loss_sum / count), "lr": lr,
                   "steps_per_sec": (step - last_step) / (now - last_time),
                   "data_wait": data_time / (now - last_time)}
            loss_sum, count, last_time, last_step, data_time = 0.0, 0, now, step, 0.0
            if is_main():
                print(json.dumps(log), flush=True)
                if wandb:
                    wandb.log(log, step=step)
        if step % cfg.val_every == 0 or step % cfg.ckpt_every == 0 or step == cfg.num_steps:
            metrics, val_scores = val.evaluate(controller, label_scale, cfg)
            if is_main():
                record = {"step": step, **metrics}
                print(json.dumps(record), flush=True)
                with open(log_file, "a") as f:
                    f.write(json.dumps(record) + "\n")
                if wandb:
                    wandb.log(metrics, step=step)
                if step % cfg.ckpt_every == 0 or step == cfg.num_steps:
                    save(metrics, val_scores)
            last_time = time.time()  # do not count validation in the throughput
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
