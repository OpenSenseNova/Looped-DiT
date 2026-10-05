<p align="center">
  <img src="assets/method.png" alt="Looped-DiT method: shared middle blocks looped N times, with Self-Modulating Attention and Deep Supervision" />
</p>

<h2 align="center">Looped-DiT: Looped Diffusion Transformer</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2609.40305"><img src="https://img.shields.io/badge/arXiv-2609.40305-b31b1b?logo=arxiv" alt="arXiv: 2609.40305" /></a>
  &nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow" alt="License: MIT" /></a>
</p>

Official PyTorch implementation of **Looped-DiT**.

Looped-DiT scales the computation of a text-to-image diffusion transformer by running a
shared group of transformer blocks several times within each denoising step: the model
gets deeper without getting larger. Naive looping does not reliably help, so Looped-DiT
adds two components. **Deep Supervision** decodes the state after every loop through the
post-loop blocks and trains each of these predictions on the same clean-image target.
**Self-Modulating Attention** (exclusive self-attention, XSA, or a head-wise attention
gate) regulates the attention updates inside the loop. The backbone is the pixel-space
MMDiT of [MiniT2I](https://github.com/PeppaKing8/minit2i-jax), conditioned on a frozen
FLAN-T5-Large.

This repository includes:
- **Training** code for Looped-DiT B/32, B/16 and L/16 (pretraining and fine-tuning).
- **Inference** at any loop depth.
- **Evaluation** on GenEval, DPG-Bench, PRISM-Bench, T2I-CoReBench, SpatialGenEval and TIIF-Short.
- **Data preparation** scripts for every training set.

## Table of Contents

- [Model Zoo](#model-zoo)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Inference](#inference)
  - [Recommended Inference Settings](#recommended-inference-settings)
- [Evaluation](#evaluation)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Evaluate Checkpoints](#evaluate-checkpoints)
- [Full Training](#full-training)
  - [Configurations](#configurations)
  - [Dataset Preparation](#dataset-preparation)
  - [Pretraining](#pretraining)
  - [Fine-Tuning](#fine-tuning)
  - [Ablation Toggles](#ablation-toggles)
- [Acknowledgments](#acknowledgments)

## Model Zoo

| Model | Patch | GenEval | DPG | PRISM | CoRe | Spatial | TIIF | Avg | Checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Looped-DiT B/32 | 32 | 85.1 | 85.3 | 54.4 | 44.5 | 52.3 | 76.1 | 66.3 | [sensenova/Looped-DiT-B32](https://huggingface.co/sensenova/Looped-DiT-B32) |
| Looped-DiT B/16 | 16 | 87.4 | 87.0 | 67.0 | 53.5 | 54.6 | 79.7 | 71.5 | [sensenova/Looped-DiT-B16](https://huggingface.co/sensenova/Looped-DiT-B16) |

Scores use the EMA weights, 100 Euler steps, guidance 6.0 and loop depth 4.

Spatial denotes SpatialGenEval. TIIF denotes the short-prompt version of TIIF-Bench.

## Repository Layout

```text
.
├── configs/          # training configs (B/32, B/16, L/16) and the evaluation config
├── looped_dit/       # model, training, sampling
│   └── eval/         # benchmark image generation and scoring
├── tools/            # dataset preparation
├── tests/            # unit tests
└── assets/
```

## Installation

Create a Python environment with a CUDA build of PyTorch (2.1 or newer), then install the
dependencies:

```bash
pip install -r requirements.txt
```

Run all commands from the repository root. Configs refer to `${DATA_ROOT}` (training data),
`${ASSET_ROOT}` (benchmark files) and `${OUTPUT_ROOT}` (checkpoints and results in the
evaluation config). They default to `data/`, `eval_assets/` and `outputs/` in the
repository and can be set as environment variables.

Sanity-check the environment with the unit tests (CPU, under a minute):

```bash
python -m pytest tests
```

## Inference

Download a checkpoint from the [Model Zoo](#model-zoo), then generate. `--loops` sets the
loop depth, and several depths give one row each:

```bash
hf download sensenova/Looped-DiT-B16 looped-dit-b16.pt --local-dir checkpoints

python -m looped_dit.sample --checkpoint checkpoints/looped-dit-b16.pt \
    --prompt "a red cube on top of a blue sphere" --out sample.png

python -m looped_dit.sample --checkpoint checkpoints/looped-dit-b16.pt \
    --prompt "a red cube on top of a blue sphere" --loops 1 2 3 4 --out loops.png
```

From Python:

```python
import torch
from looped_dit.pipeline import TextEncoder, generate, load_model

device = torch.device("cuda")
model, cfg = load_model("checkpoints/looped-dit-b16.pt", device)  # EMA weights, bf16
text_encoder = TextEncoder(cfg.text_encoder, cfg.prompt_length, device)
torch.manual_seed(0)
image = generate(model, text_encoder, ["a red cube on top of a blue sphere"], num_loops=4)[0]
image.save("sample.png")
```

### Recommended Inference Settings

These are the settings of the paper's main results and the defaults of the scripts.

| Setting | Value |
| --- | --- |
| Sampler | Euler, 100 steps |
| Classifier-free guidance scale | 6.0 |
| Loop depth | 4, the trained depth (other depths work without retraining) |
| Weights / precision | EMA, bfloat16 |
| Resolution | 512 x 512 |

## Evaluation

### Evaluation Metrics

This repository does not include benchmark data. Copy the files below from the benchmark
repositories into `${ASSET_ROOT}`. The paths are the defaults of `configs/eval.yml`, and the
folder of the benchmark repository a file comes from is in parentheses. The judge and
scorer models (Qwen, mPLUG, CLIP) are downloaded on first use, but the Mask2Former weights
are not.

| Benchmark | Assets | Scorer | Extra packages |
| --- | --- | --- | --- |
| [GenEval](https://github.com/djghosh13/geneval) | `geneval/evaluation_metadata.jsonl` (from `prompts/`), Mask2Former weights in `geneval/detector/` (from `evaluation/download_models.sh`) | Mask2Former + CLIP | `mmdet` 2.x, `mmcv-full`, `open_clip_torch`, `clip-benchmark` |
| [DPG-Bench](https://huggingface.co/datasets/Jialuo21/DPG-Bench) | loaded from the Hugging Face Hub | mPLUG-large VQA | `modelscope` |
| [PRISM-Bench](https://github.com/rongyaofang/prism-bench) | `prism/captions/en/`, `prism/eval_qwen25.py` (from `evaluation/`) | Qwen2.5-VL-72B-Instruct | `vllm`, `qwen-vl-utils` |
| [T2I-CoReBench](https://github.com/KlingAIResearch/T2I-CoReBench) | `corebench/data/` | Qwen3-VL-32B-Thinking | `vllm`, `qwen-vl-utils` |
| [SpatialGenEval](https://github.com/AMAP-ML/SpatialGenEval) | `spatial_geneval/SpatialGenEval_T2I_Prompts.jsonl` (from `eval/`) | Qwen2.5-VL-72B-Instruct, 5 rollouts, 4/5 majority | `vllm`, `qwen-vl-utils` |
| [TIIF-Bench](https://github.com/A113N-W3I/TIIF-Bench) (short prompts) | `tiif/data/`, `tiif/eval_with_vlm.py` (from `eval/`) | GPT-4o through the OpenAI API | `requests` |

The Qwen judges run locally with vLLM on 8 GPUs (set `tensor_parallel_size` under a
benchmark to change this). GenEval (mmdet) and the judges (vLLM) usually need their own
Python environments. Set them under `python:` in the eval config. We used mmdet 2.28.2 with
mmcv-full 1.7.2 for GenEval, and vLLM 0.24 with transformers 5.12 and qwen-vl-utils 0.0.14
for the judges. TIIF-Bench is judged by GPT-4o: set `OPENAI_API_KEY` (`api_base` in the eval
config accepts any OpenAI-compatible endpoint).
The GenEval scorer looks for the Mask2Former config in an mmdetection 2.x source checkout,
as in the GenEval instructions. With a pip-installed `mmdet`, set `detector_config`.

### Evaluate Checkpoints

Set `checkpoint` and `output_dir` in `configs/eval.yml`, then run:

```bash
python -m looped_dit.eval.run --config configs/eval.yml
```

This generates the images of all six benchmarks on 8 GPUs, scores them and prints a table
(`scores.json` in the output directory). Finished benchmarks are skipped on a rerun. The
noise seed of a prompt depends on the GPU that renders it, so use 8 GPUs to reproduce a
run exactly. Each step can also be run on its own, see `looped_dit/eval/`.

## Full Training

### Configurations

| Model | Patch | Blocks (pre-loop, looped, post-loop) | Pretraining | Fine-tuning | Configs |
| --- | ---: | :---: | --- | --- | --- |
| Looped-DiT B/32 | 32 | [6,5,6] | CC12M, 250k steps | Std-120K, 40k steps | `configs/b32_*.yml` |
| Looped-DiT B/16 | 16 | [6,5,6] | CC12M + FLUX-Reason-6M, 500k steps | Std-120K + Fine-T2I, 80k steps | `configs/b16_*.yml` |
| Looped-DiT L/16 | 16 | [8,7,8] | CC12M + FLUX-Reason-6M, 500k steps | Std-120K + Fine-T2I, 80k steps | `configs/l16_*.yml` |

All models generate 512 x 512 images with a frozen FLAN-T5-Large text encoder. The looped
blocks run 4 times, so B/32 and B/16 apply 32 blocks per forward pass with 17 distinct
blocks (L/16: 44 with 23). The configs train with deep supervision (Final + Mean
weighting) and XSA.

### Dataset Preparation

Training data goes into `${DATA_ROOT}`:

```text
data/
├── cc12m_chunks/         # pretraining tensor chunks (chunk_*.pt)
├── fluxreason_chunks/
└── finetune/             # WebDataset shards of (jpg or png, txt) pairs
    ├── blip3o_60k/  dalle3/  sharegpt4o/  fine_t2i/
```

The pretraining chunks hold decoded 512 x 512 images, about 0.8 GB per 1,024 samples:
roughly 5 TB each for CC12M and FLUX-Reason-6M.

**CC12M.** Pretraining uses CC12M with the LLaVA-NeXT captions of
[CaptionEmporium/conceptual-captions-cc12m-llavanext](https://huggingface.co/datasets/CaptionEmporium/conceptual-captions-cc12m-llavanext).
Download the images with [img2dataset](https://github.com/rom1504/img2dataset), installed
in a separate environment (it requires an older `webdataset` than this repository), then
write tensor chunks:

```bash
hf download CaptionEmporium/conceptual-captions-cc12m-llavanext --repo-type dataset \
    --include "*.jsonl.gz" --local-dir data/cc12m_meta
python -c "import pandas as pd; df = pd.read_json('data/cc12m_meta/train.jsonl.gz', lines=True); \
df[df.status == 'success'][['url', 'caption_llava']].to_parquet('data/cc12m_meta/urls.parquet')"
img2dataset --url_list data/cc12m_meta/urls.parquet --input_format parquet \
    --url_col url --caption_col caption_llava --output_format webdataset \
    --output_folder data/cc12m_wds --image_size 512 --resize_mode center_crop \
    --encode_format jpg --number_sample_per_shard 10000 --processes_count 16 --thread_count 64
python tools/make_chunks.py --webdataset data/cc12m_wds --out data/cc12m_chunks
```

**FLUX-Reason-6M.** B/16 and L/16 also pretrain on
[FLUX-Reason-6M](https://huggingface.co/datasets/LucasFang/FLUX-Reason-6M) images with the
dense captions released with [i1](https://huggingface.co/datasets/zlab-princeton/i1-captions):
five captions per image, of which `make_chunks.py` keeps one, chosen at random but the same
on every run. Download the FLUX-Reason part of the captions (10 GB), then point `--captions`
at it:

```bash
hf download zlab-princeton/i1-captions --repo-type dataset --include "fluxreason/*" \
    --local-dir data/i1-captions
python tools/make_chunks.py --hf-dataset LucasFang/FLUX-Reason-6M --id-column id \
    --captions data/i1-captions/fluxreason --out data/fluxreason_chunks
```

**Fine-tuning data.** Std-120K mixes
[BLIP3o-60K](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k),
[DALL-E 3](https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset) and the
text-to-image part of [ShareGPT-4o-Image](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image)
with weights 0.06 / 0.016 / 0.04. B/16 and L/16 add
[Fine-T2I](https://huggingface.co/datasets/ma-xu/fine-t2i) for half of the samples:

```bash
python tools/prepare_finetune_data.py --out data/finetune                      # Std-120K
python tools/prepare_finetune_data.py --out data/finetune --sources fine_t2i   # Fine-T2I
```

### Pretraining

Launch on every node with `torchrun`. The global batch is 1024: each step accumulates
gradients over 1024 / (`micro_batch_size` x GPUs) micro-batches, so that product must divide
1024 (change `micro_batch_size` with `--set` if it does not). A run resumes from the newest
checkpoint in its output directory.

```bash
torchrun --nnodes 2 --nproc_per_node 8 --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR --master_port 29500 \
    -m looped_dit.train --config configs/b32_pretrain.yml --output-dir outputs/b32_pretrain
```

The reference setups are 2 nodes for B/32, 4 for B/16 and 8 for L/16 (8 GPUs with 80 GB
each). Each data-loader worker keeps a shuffle buffer of 4,096 decoded images (about 3 GB),
so pretraining needs about 150 GB of host memory per 8-GPU node. Lower `shuffle_buffer` or
`num_workers` if that is too much. Add `--wandb` to log to Weights & Biases.

| | B/32 | B/16 | L/16 |
| --- | --- | --- | --- |
| Steps | 250k | 500k | 500k |
| Learning rate (after 5k warmup) | 4e-4 | 4e-4 | 2e-4 |
| EMA decay | 0.99995 | 0.99995 | 0.9999 |

All models use AdamW with betas (0.9, 0.95) and no weight decay, gradient clipping at 0.1,
bf16 autocast, noise scale 2.0, logit-normal timesteps (mean -0.8, std 0.8) and 10% prompt
dropout.

### Fine-Tuning

Fine-tuning starts from the pretrained weights, EMA and optimizer state and continues the
step counter, without warmup (learning rate 4e-4 and EMA 0.99995 for every model):

```bash
torchrun ... -m looped_dit.train --config configs/b32_finetune.yml --output-dir outputs/b32_finetune \
    --init-from outputs/b32_pretrain/checkpoints/checkpoint_0250000.pt
```

### Ablation Toggles

Each Looped-DiT component is a config switch, so the ablations of the paper are config edits
(or command-line overrides such as `--set use_xsa=false use_attn_gate=true`). Each row lists
only the switches to change. Everything else stays as in the shipped configs, which train with
deep supervision (Final + Mean) and XSA. For example, the gated-attention row keeps deep
supervision on, so add `deep_supervision: false` to train the gate alone:

| Model | Config |
| --- | --- |
| MiniT2I baseline without looping (the same 17 blocks, each run once) | `num_loops: 1`, `deep_supervision: false`, `use_xsa: false` |
| Deeper MiniT2I (32 blocks without weight sharing, compute-matched) | `share_loop_weights: false`, `deep_supervision: false`, `use_xsa: false` |
| Naive looping | `deep_supervision: false`, `use_xsa: false` |
| Other deep-supervision weightings (the paper's Exponential, or uniform) | `deep_supervision_weighting: exponential` or `uniform` |
| Gated attention instead of XSA | `use_xsa: false`, `use_attn_gate: true` |

## Acknowledgments

This codebase builds on:

- [MiniT2I](https://github.com/PeppaKing8/minit2i-jax) and its [PyTorch implementation](https://github.com/Hope7Happiness/minit2i-torch): the backbone, data pipeline and training recipe.
- [GenEval](https://github.com/djghosh13/geneval): the object-detection scorer in `looped_dit/eval/geneval/`, adapted with a progress bar and a JSON summary (MIT license).
- The authors of the datasets and benchmarks linked above.

This project is released under the [MIT License](LICENSE).
