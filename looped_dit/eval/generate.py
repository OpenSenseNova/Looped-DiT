"""Generate the images of one benchmark; launch with torchrun to split prompts over GPUs.

    torchrun --nproc_per_node=8 -m looped_dit.eval.generate --benchmark geneval \
        --data eval_assets/geneval/evaluation_metadata.jsonl --checkpoint CKPT --outdir OUT/geneval_images

With --controller and --target-loops (or --threshold) the loop depth is adaptive
(looped_dit/adaptive); the loop counts are written to OUTDIR/.loops/.

Images that already exist are skipped, so an interrupted run can be restarted.
The noise seed of a prompt depends on its index and on the rank that renders it:
use the same number of GPUs to reproduce a run exactly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

from ..adaptive import eval as adaptive
from ..pipeline import TextEncoder, generate, load_model
from ..utils import init_distributed, rank, world_size
from .benchmarks import BENCHMARKS, Prompt, image_path, load_prompts


def output_paths(benchmark: str, outdir: Path, prompt: Prompt, samples: int) -> list[Path]:
    if benchmark == "geneval":
        return [outdir / prompt.item_id / "samples" / f"{i:05d}.png" for i in range(samples)]
    if benchmark == "dpg":
        return [outdir / f"{prompt.item_id}.png"]
    return [image_path(outdir, prompt, i) for i in range(samples)]


def save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    image.save(tmp, format="PNG")
    os.replace(tmp, path)


def grid_2x2(images: list[Image.Image]) -> Image.Image:
    w, h = images[0].size
    canvas = Image.new("RGB", (2 * w, 2 * h))
    for image, xy in zip(images, [(0, 0), (w, 0), (0, h), (w, h)]):
        canvas.paste(image, xy)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark images.")
    parser.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    parser.add_argument("--data", required=True, help="benchmark prompt file or directory")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=6.0)
    parser.add_argument("--loops", type=int, default=None, help="loop depth (default: as trained)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text-encoder", help="override the text encoder path from the checkpoint")
    adaptive.add_arguments(parser)
    args = parser.parse_args()
    adaptive.check_arguments(parser, args)
    if args.benchmark == "dpg" and args.samples_per_prompt != 4:
        parser.error("DPG-Bench is scored on 2x2 grids: use --samples-per-prompt 4")

    device = init_distributed()
    torch.backends.cuda.matmul.allow_tf32 = True  # as in training; here it also rounds the fp32 T5 encoder
    torch.backends.cudnn.allow_tf32 = True
    outdir = Path(args.outdir)
    prompts = load_prompts(args.benchmark, args.data)[rank() :: world_size()]
    todo = [p for p in prompts if not all(x.exists() for x in output_paths(args.benchmark, outdir, p, args.samples_per_prompt))
            or (args.controller and not adaptive.loops_path(outdir, p.index).exists())]
    print(f"[rank {rank()}] {args.benchmark}: {len(todo)} of {len(prompts)} prompts to generate", flush=True)
    if todo:
        model, cfg = load_model(args.checkpoint, device)
        text_encoder = TextEncoder(args.text_encoder or cfg.text_encoder, cfg.prompt_length, device)
        adaptive_generate = None
        if args.controller:
            adaptive_generate = adaptive.AdaptiveGenerator(args.controller, model, text_encoder, device, cfg.image_size,
                                                           cfg.noise_scale, args.target_loops, args.threshold)
        for n, prompt in enumerate(todo, 1):
            images: list[Image.Image] = []
            while len(images) < args.samples_per_prompt:
                batch = min(args.batch_size, args.samples_per_prompt - len(images))
                torch.manual_seed(args.seed + prompt.index * 1000 + len(images) + rank() * 1_000_000)
                if adaptive_generate:
                    images += adaptive_generate([prompt.text] * batch, args.steps, args.cfg_scale)
                else:
                    images += generate(model, text_encoder, [prompt.text] * batch, cfg.image_size, args.steps,
                                       args.cfg_scale, args.loops, cfg.noise_scale)
            paths = output_paths(args.benchmark, outdir, prompt, args.samples_per_prompt)
            if args.benchmark == "dpg":
                images = [grid_2x2(images)]
            if args.benchmark == "geneval":
                (outdir / prompt.item_id).mkdir(parents=True, exist_ok=True)
                (outdir / prompt.item_id / "metadata.jsonl").write_text(json.dumps(prompt.metadata) + "\n")
            for image, path in zip(images, paths):
                save(image, path)
            if adaptive_generate:
                adaptive_generate.save_loops(outdir, prompt.index, prompt.item_id)
            if n % 50 == 0 or n == len(todo):
                print(f"[rank {rank()}] {n}/{len(todo)}", flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
