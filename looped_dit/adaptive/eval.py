"""Adaptive looping in benchmark generation (looped_dit/eval/generate.py) and sample.py.

With --controller and --target-loops (or --threshold) the loop depth is adaptive. The
loop count of every image and denoising step is written next to the images, to
OUTDIR/.loops/<prompt index>.json, and loop_summary averages them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image

from ..utils import rank
from .controller import load_controller, resolve_threshold
from .sampler import generate_adaptive


def add_arguments(parser: argparse.ArgumentParser, several: bool = False) -> None:
    """--controller, --target-loops, --threshold (several values each with several=True)."""
    nargs = "+" if several else None
    parser.add_argument("--controller", help="adaptive looping: a calibrated controller checkpoint")
    parser.add_argument("--target-loops", type=float, nargs=nargs, help="adaptive: calibrated average loop count")
    parser.add_argument("--threshold", type=float, nargs=nargs, help="adaptive: controller threshold")


def check_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace, several: bool = False) -> None:
    budgets = [x for x in (args.target_loops, args.threshold) if x is not None]
    if args.controller and (not budgets or (not several and len(budgets) > 1)):
        parser.error("--controller needs " + ("--target-loops and/or --threshold" if several else
                                              "one of --target-loops and --threshold"))
    if budgets and not args.controller:
        parser.error("--target-loops / --threshold need --controller")
    if args.controller and getattr(args, "loops", None) not in (None, [None]):
        parser.error("--loops and --controller exclude each other")


def loops_path(outdir: Path, index: int) -> Path:
    return Path(outdir) / ".loops" / f"{index:06d}.json"


def loop_summary(outdir: str | Path) -> dict | None:
    """Average loop count of an adaptive run (over images and denoising steps), or None."""
    files = sorted((Path(outdir) / ".loops").glob("*.json"))
    if not files:
        return None
    per_image = [sum(map(int, s)) / len(s) for f in files for s in json.loads(f.read_text())["loops"]]
    return {"images": len(per_image), "mean_loops": sum(per_image) / len(per_image)}


class AdaptiveGenerator:
    """generate() with adaptive loop depth that keeps the loop counts of the images it makes."""

    def __init__(self, controller_path: str, model, text_encoder, device, image_size: int, noise_scale: float,
                 target_loops: float | None = None, threshold: float | None = None):
        self.model, self.text_encoder = model, text_encoder
        self.image_size, self.noise_scale = image_size, noise_scale
        self.controller, ckpt = load_controller(controller_path, device)
        self.threshold = resolve_threshold(ckpt, target_loops, threshold)
        self.loops: list[str] = []
        print(f"[rank {rank()}] adaptive looping, threshold {self.threshold:.6g}", flush=True)

    def __call__(self, prompts: list[str], steps: int, cfg_scale: float) -> list[Image.Image]:
        images, counts = generate_adaptive(self.model, self.controller, self.text_encoder, prompts, self.threshold,
                                           self.image_size, steps, cfg_scale, self.noise_scale)
        self.loops += ["".join(map(str, row.tolist())) for row in counts]  # one digit per denoising step
        return images

    def save_loops(self, outdir: Path, index: int, item_id: str) -> None:
        """Write the loop counts collected since the last call (one prompt's images)."""
        path = loops_path(outdir, index)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"item_id": item_id, "threshold": self.threshold, "loops": self.loops}))
        os.replace(tmp, path)
        self.loops = []
