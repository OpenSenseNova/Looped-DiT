"""Run the benchmark suite for one checkpoint: generate images, score them, print a table.

    python -m looped_dit.eval.run --config configs/eval.yml

Each benchmark is generated with torchrun over `gpus` GPUs, then scored by its own
script (GenEval: Mask2Former + CLIP, DPG-Bench: mPLUG, PRISM / CoReBench /
SpatialGenEval: a vLLM judge, TIIF-Bench: GPT-4o). Finished benchmarks are skipped on a rerun.

With `controller` and `target_loops` (or `threshold`) set, sampling uses adaptive
looping (looped_dit/adaptive) and scores.json also reports the measured average loops.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from ..config import REPO_ROOT, load_yaml

SAMPLES_PER_PROMPT = {"geneval": 4, "dpg": 4, "prism": 1, "corebench": 4, "spatial_geneval": 1, "tiif": 1}
JUDGE_OPTIONS = ("judge_model", "tensor_parallel_size", "batch_size", "max_model_len", "max_new_tokens", "max_failed", "retries")
GENEVAL_DIR = Path(__file__).resolve().parent / "geneval"


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))}
    subprocess.run(cmd, env=env, check=True)


def flags(options: dict, names: tuple[str, ...]) -> list[str]:
    return [arg for name in names if options.get(name) is not None for arg in (f"--{name.replace('_', '-')}", str(options[name]))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Looped-DiT checkpoint on the benchmark suite.")
    parser.add_argument("--config", required=True)
    cfg = load_yaml(parser.parse_args().config)
    out = Path(cfg["output_dir"])
    # Results are reused on a rerun, so an output directory belongs to one checkpoint and sampling setup.
    sampling = ("steps", "cfg_scale", "loops", "seed", "text_encoder", "controller", "target_loops", "threshold")
    settings = {key: cfg.get(key) for key in ("checkpoint", "gpus", *sampling)}
    record = out / "settings.json"
    if record.exists() and json.loads(record.read_text()) != settings:
        raise SystemExit(f"{out} holds results of other settings (see {record}); choose a new output_dir")
    out.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(settings, indent=2))
    python = {"geneval": sys.executable, "dpg": sys.executable, "judge": sys.executable, **(cfg.get("python") or {})}
    gpus = int(cfg.get("gpus", 1))
    launcher = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpus}"] if gpus > 1 else [sys.executable]

    scores, loops = {}, {}
    for name, bench in cfg["benchmarks"].items():
        images, results = out / f"{name}_images", out / f"{name}_results"
        summary = results / "summary.json"
        samples = int(bench.get("samples_per_prompt", SAMPLES_PER_PROMPT[name]))
        if not summary.exists():
            run(launcher + ["-m", "looped_dit.eval.generate", "--benchmark", name, "--data", str(bench["data"]),
                            "--checkpoint", str(cfg["checkpoint"]), "--outdir", str(images),
                            "--samples-per-prompt", str(samples)]
                + flags(cfg, sampling))
            results.mkdir(parents=True, exist_ok=True)
            if name == "geneval":
                run([python["geneval"], str(GENEVAL_DIR / "evaluate_images.py"), str(images),
                     "--outfile", str(results / "results.jsonl"), "--model-path", str(bench["detector"])]
                    + (["--model-config", str(bench["detector_config"])] if bench.get("detector_config") else []))
                run([python["geneval"], str(GENEVAL_DIR / "summary_scores.py"), str(results / "results.jsonl"), str(summary)])
            elif name == "dpg":
                run([python["dpg"], "-m", "looped_dit.eval.dpg", "--image-dir", str(images), "--data", str(bench["data"]),
                     "--out-dir", str(results)] + flags(bench, ("mplug",)))
            elif name == "tiif":  # judged through an OpenAI-compatible API
                run([sys.executable, "-m", "looped_dit.eval.tiif", "--data", str(bench["data"]),
                     "--eval-script", str(bench["script"]), "--image-dir", str(images), "--out-dir", str(results),
                     "--samples-per-prompt", str(samples)]
                    + flags(bench, ("judge_model", "api_base", "api_key_env", "max_workers", "max_failed")))
            elif name == "spatial_geneval":
                run([python["judge"], "-m", "looped_dit.eval.spatial_geneval", "--data", str(bench["data"]),
                     "--image-dir", str(images), "--out-dir", str(results), "--samples-per-prompt", str(samples)]
                    + flags(bench, ("judge_model", "tensor_parallel_size", "batch_size", "max_model_len", "max_new_tokens")))
            else:
                run([python["judge"], "-m", "looped_dit.eval.judge", "--benchmark", name, "--data", str(bench["data"]),
                     "--image-dir", str(images), "--out-dir", str(results), "--samples-per-prompt", str(samples)]
                    + (["--prism-script", str(bench["script"])] if name == "prism" else []) + flags(bench, JUDGE_OPTIONS))
        result = json.loads(summary.read_text())
        scores[name] = 100 * result["overall_score"] if name == "geneval" else result["overall"]
        if cfg.get("controller"):
            from ..adaptive.eval import loop_summary

            loops[name] = loop_summary(images)

    report = dict(scores)
    if loops:  # average loops per denoising step, per benchmark and over all images
        count = sum(v["images"] for v in loops.values())
        loops["all"] = {"images": count, "mean_loops": sum(v["images"] * v["mean_loops"] for v in loops.values()) / count}
        report["mean_loops"] = {name: v["mean_loops"] for name, v in loops.items()}
    (out / "scores.json").write_text(json.dumps(report, indent=2))
    print("\n".join(f"{name:>16}  {score:6.2f}" + (f"  ({loops[name]['mean_loops']:.3f} loops)" if name in loops else "")
                    for name, score in scores.items()))
    if loops:
        print(f"{'average loops':>16}  {loops['all']['mean_loops']:.3f}")


if __name__ == "__main__":
    main()
