#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_DnRv3_fixed import MixtureGeneratorDnRv3, filelist_generate
from bin._manifest_utils import read_jsonl

try:
    import torch.distributed as dist
except Exception:
    dist = None


class ManifestMixtureGenerator(MixtureGeneratorDnRv3):
    def __init__(
        self,
        manifest_dir,
        split,
        mixture_length=60.0,
        sampling_rate=16000,
        peak_norm_db=-2.0,
        rank=0,
        enable_background_sfx=False,
    ):
        self.seq_dur = mixture_length
        self.sr = sampling_rate
        self.rank = rank
        self.background = enable_background_sfx
        self.true_peak_db = peak_norm_db
        self.partition = split
        self.without_replacement = False
        self.source_usage_tracker = {"music": 0, "speech": 0, "sfx": 0}
        self.mu_ref = -27
        self.mu_mix = -27
        self.sigma_mix = 1
        self.delta_mu_track = {"music": -5, "sfx": -5, "speech": 0, "background": -13}
        self.sigma_track = {"music": 6, "sfx": 6, "speech": 4, "background": 6}
        self.sigma_event = {"music": 10, "sfx": 10, "speech": 6, "background": 10}
        self.lamdb_event = {
            "music": 7,
            "sfx": 12 if split == "train" else 24,
            "speech": 12 if split == "train" else 24,
            "background": 24,
        }
        self.overlap_gamma = {
            "music": 1.0,
            "sfx": 0.5 if split == "train" else 1.0,
            "speech": 0.75 if split == "train" else 1.0,
            "background": 0.0,
        }
        self.beta_min = {"music": 0.3, "sfx": 0.3, "speech": 1.0, "background": 0.3}
        self.T_min = {"music": 0, "sfx": 0.5, "speech": 0, "background": 1.0}
        self.files = {}
        manifest_split_dir = Path(manifest_dir) / split
        for stem in ["speech", "music", "sfx"]:
            rows = read_jsonl(manifest_split_dir / f"{stem}.jsonl")
            self.files[stem] = [{"files": row["path"]} for row in rows]
        if enable_background_sfx:
            bg_manifest = manifest_split_dir / "background.jsonl"
            if not bg_manifest.exists():
                raise FileNotFoundError(f"Background manifest missing: {bg_manifest}")
            bg_rows = read_jsonl(bg_manifest)
            self.files["background"] = [{"files": row["path"]} for row in bg_rows]


def get_rank_world_size():
    if dist is None:
        return 0, 1, False
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), True
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        return dist.get_rank(), dist.get_world_size(), True
    return 0, 1, False


def write_provenance(output_root, dataset_name, split, args, generator):
    info = {
        "dataset_name": dataset_name,
        "split": split,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "mixture_length": args.mixture_length,
        "enable_background_sfx": args.enable_background_sfx,
        "manifest_dir": args.manifest_dir,
        "output_root": args.output_root,
        "source_counts": {key: len(value) for key, value in generator.files.items()},
    }
    prov_dir = Path(output_root) / dataset_name / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    (prov_dir / f"{split}.json").write_text(json.dumps(info, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate AVDnR splits from explicit manifests.")
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-name", default="AVDnR")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--mixture-length", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=20260309)
    parser.add_argument("--enable-background-sfx", action="store_true")
    args = parser.parse_args()

    rank, world_size, distributed = get_rank_world_size()
    np.random.seed(args.seed + rank)
    generator = ManifestMixtureGenerator(
        manifest_dir=args.manifest_dir,
        split=args.split,
        mixture_length=args.mixture_length,
        rank=rank,
        enable_background_sfx=args.enable_background_sfx,
    )
    write_provenance(args.output_root, args.dataset_name, args.split, args, generator)

    output_dir = Path(args.output_root) / args.dataset_name / args.split / f"parts_{rank}"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    failures = 0
    while generated < args.num_samples:
        sample = generator()
        if sample is None:
            failures += 1
            continue
        normalized_sources, mixture, annots = sample
        sample_dir = output_dir / str(generated)
        sample_dir.mkdir(parents=True, exist_ok=True)
        for stem, audio in normalized_sources.items():
            sf.write(sample_dir / f"{stem}.wav", audio, 16000)
        sf.write(sample_dir / "mixture.wav", mixture, 16000)
        (sample_dir / "annots.json").write_text(json.dumps(annots, indent=2) + "\n")
        generated += 1
        if generated % 50 == 0:
            print(f"rank={rank} generated={generated}/{args.num_samples} failures={failures}", flush=True)

    if distributed:
        dist.barrier()
    if rank == 0:
        filelist_generate(args.output_root, args.dataset_name, args.split)
        print(f"Completed split={args.split} into {Path(args.output_root) / args.dataset_name / args.split}", flush=True)


if __name__ == "__main__":
    main()
