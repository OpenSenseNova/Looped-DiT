"""Data of the controller: the training streams of looped_dit/data.py split by caption.

A fixed share of the captions (by hash) is held out: the controller never trains on
them, and its validation set and calibration prompts come from them. The split is
applied before images are decoded, so drawing from the small held-out side is cheap.
The WebDataset shard order comes from the seeded generator, so a controller seed
fixes the data order.
"""

from __future__ import annotations

import random
import zlib

import torch
import webdataset as wds
from torch.utils.data import DataLoader

from ..config import TrainConfig
from ..data import ChunkStream, WebDatasetMix, collate, worker_seed, worker_shard


class HoldOut:
    """Deterministic caption split: True for captions whose hash falls below `fraction`
    (held_out=True) or above it (held_out=False). Takes a caption or the prompt's token ids."""

    def __init__(self, fraction: float, held_out: bool = True):
        self.fraction, self.held_out = fraction, held_out

    def __call__(self, caption: str | torch.Tensor) -> bool:
        if isinstance(caption, str):
            key = caption.strip().encode("utf-8")
        else:
            key = caption.to(torch.int32).numpy().tobytes()
        return (zlib.crc32(key) < self.fraction * 2**32) == self.held_out


class SplitChunkStream(ChunkStream):
    """ChunkStream restricted to one side of a HoldOut split (judged on the prompt's token ids)."""

    def __init__(self, cfg: TrainConfig, seed_offset: int, keep: HoldOut):
        super().__init__(cfg, seed_offset)
        self.keep = keep

    def samples(self, files, rng: random.Random):
        while True:
            for path in rng.sample(files, len(files)):
                chunk = torch.load(path, map_location="cpu", weights_only=False)
                for i in rng.sample(range(len(chunk["input_ids"])), len(chunk["input_ids"])):
                    if not self.keep(chunk["input_ids"][i][chunk["attention_mask"][i].bool()]):
                        continue
                    yield {
                        "pixel_values": chunk["pixel_values"][i].clone(),
                        "input_ids": chunk["input_ids"][i].long(),
                        "attention_mask": chunk["attention_mask"][i].long(),
                    }


class SplitWebDatasetMix(WebDatasetMix):
    """WebDatasetMix restricted to one side of a HoldOut split (judged on the caption)."""

    def __init__(self, cfg: TrainConfig, seed_offset: int, keep: HoldOut):
        super().__init__(cfg, seed_offset)
        self.keep = keep

    @staticmethod
    def caption(sample: dict) -> str:
        caption = sample.get("txt", "")
        return caption.decode("utf-8", errors="replace") if isinstance(caption, bytes) else str(caption)

    def source(self, index: int, epoch: int):
        """One pass over this worker's shards of source `index`, in seeded order."""
        urls = list(worker_shard(self.shards[index], min_one=True))
        rng = random.Random(worker_seed(self.cfg, self.seed_offset + 1000 * epoch + index))
        rng.shuffle(urls)
        samples = wds.WebDataset(
            urls,
            handler=wds.warn_and_continue,
            empty_check=False,
            shardshuffle=False,
            nodesplitter=lambda src: src,
            workersplitter=lambda src: src,
        ).select(lambda sample: self.keep(self.caption(sample)))
        return (
            samples.shuffle(self.cfg.shuffle_buffer, rng=rng)
            .decode("pil", handler=wds.warn_and_continue)
            .map(self.to_sample, handler=wds.warn_and_continue)
        )


def split_loader(data_cfg: TrainConfig, holdout_fraction: float, held_out: bool, batch: int,
                 seed_offset: int = 0, workers: int | None = None) -> DataLoader:
    """Batches of `batch` images from one side of the split (held_out: the held-out captions)."""
    keep = HoldOut(holdout_fraction, held_out)
    stream_type = SplitChunkStream if data_cfg.dataset == "chunks" else SplitWebDatasetMix
    workers = data_cfg.num_workers if workers is None else workers
    return DataLoader(stream_type(data_cfg, seed_offset, keep), batch_size=batch, num_workers=workers,
                      collate_fn=collate, pin_memory=True, drop_last=True, persistent_workers=workers > 0,
                      prefetch_factor=data_cfg.prefetch_factor if workers > 0 else None)
