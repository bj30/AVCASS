#!/usr/bin/env python3
import argparse
import csv
import inspect
import json
import os
import subprocess
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import tqdm
from pesq import pesq_batch
from threadpoolctl import threadpool_limits
from torchmetrics.functional.audio import (
    scale_invariant_signal_distortion_ratio,
    scale_invariant_signal_noise_ratio,
    signal_distortion_ratio,
    signal_noise_ratio,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "av_cass" / "eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "av_cass" / "eval"))

from fad_test import create_symlinks, SimpleDataset  # noqa: E402
from audioldm_eval import EvaluationHelper  # noqa: E402


STEMS = ["speech", "sfx", "music"]


def patch_audioldm_eval_compat():
    import audioldm_eval.audio.stft as audioldm_stft
    import librosa.filters
    import librosa.util

    pad_center_signature = inspect.signature(librosa.util.pad_center)
    size_param = pad_center_signature.parameters.get("size")
    if size_param is None or size_param.kind is not inspect.Parameter.KEYWORD_ONLY:
        compat_pad_center = None
    else:
        original_pad_center = librosa.util.pad_center

        def compat_pad_center(data, size, axis=-1, **kwargs):
            return original_pad_center(data, size=size, axis=axis, **kwargs)

        librosa.util.pad_center = compat_pad_center
        audioldm_stft.pad_center = compat_pad_center

    mel_signature = inspect.signature(librosa.filters.mel)
    sr_param = mel_signature.parameters.get("sr")
    if sr_param is None or sr_param.kind is not inspect.Parameter.KEYWORD_ONLY:
        return

    original_mel = librosa.filters.mel

    def compat_mel(
        sr,
        n_fft,
        n_mels=128,
        fmin=0.0,
        fmax=None,
        htk=False,
        norm="slaney",
        dtype=np.float32,
    ):
        return original_mel(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
            norm=norm,
            dtype=dtype,
        )

    librosa.filters.mel = compat_mel
    audioldm_stft.librosa_mel_fn = compat_mel


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def prepare_symlink_roots(gt_root: Path, pred_root: Path, symlink_root: Path):
    gt_symlink_root = symlink_root / "gt_avdnr_test"
    pred_symlink_root = symlink_root / "pred_avdnr"
    ensure_clean_dir(gt_symlink_root)
    ensure_clean_dir(pred_symlink_root)
    create_symlinks(str(gt_root), str(gt_symlink_root), "avdnr_test")
    create_symlinks(str(pred_root), str(pred_symlink_root), "avdnr")
    return gt_symlink_root, pred_symlink_root


def evaluate_distribution_metrics(pred_symlink_root: Path, gt_symlink_root: Path, out_dir: Path, stems=None, device_name=None):
    stems = stems or STEMS
    device = torch.device(device_name or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    original_torch_load = torch.load
    original_cwd = Path.cwd()
    ckpt_dir = ROOT / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def safe_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        if device.type == "cpu":
            kwargs.setdefault("map_location", torch.device("cpu"))
        return original_torch_load(*args, **kwargs)

    torch.load = safe_torch_load

    try:
        os.chdir(ROOT)
        patch_audioldm_eval_compat()
        evaluator = EvaluationHelper(16000, device, backbone="cnn14")
        results = {}
        for stem in stems:
            source_path = pred_symlink_root / stem
            target_path = gt_symlink_root / stem
            metrics = evaluator.main(str(source_path), str(target_path), limit_num=None)
            results[stem] = metrics
            (out_dir / f"{stem}_distribution.json").write_text(json.dumps(metrics, indent=2) + "\n")
        return results
    finally:
        os.chdir(original_cwd)
        torch.load = original_torch_load


def evaluate_reference_metrics(pred_symlink_root: Path, gt_symlink_root: Path, out_dir: Path, stems=None, device_name=None):
    stems = stems or STEMS
    device = torch.device(device_name or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    results = {}
    for stem in stems:
        dataset = SimpleDataset(str(pred_symlink_root / stem), str(gt_symlink_root / stem), True, "avdnr", stem)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)
        n_samples = 0
        metrics_dict = {
            "sdr": 0.0,
            "snr": 0.0,
            "sisnr": 0.0,
            "sisdr": 0.0,
            "mixture_sdr": 0.0,
            "mixture_snr": 0.0,
            "mixture_sisnr": 0.0,
            "mixture_sisdr": 0.0,
        }
        rows = []
        for source_audio, target_audio, mixture_audio, source_file in tqdm.tqdm(data_loader, desc=f"Reference metrics {stem}"):
            source_audio = source_audio.to(device)
            target_audio = target_audio.to(device)
            mixture_audio = mixture_audio.to(device)
            if mixture_audio.ndim == 3:
                mixture_audio = mixture_audio.squeeze(1)
            if target_audio.ndim == 3:
                target_audio = target_audio.squeeze(1)
            if source_audio.ndim == 3:
                source_audio = source_audio.squeeze(1)

            sdr = signal_distortion_ratio(source_audio, target_audio)
            snr = signal_noise_ratio(source_audio, target_audio)
            sisnr = scale_invariant_signal_noise_ratio(source_audio, target_audio)
            sisdr = scale_invariant_signal_distortion_ratio(source_audio, target_audio)
            mixture_sdr = signal_distortion_ratio(mixture_audio, target_audio)
            mixture_snr = signal_noise_ratio(mixture_audio, target_audio)
            mixture_sisnr = scale_invariant_signal_noise_ratio(mixture_audio, target_audio)
            mixture_sisdr = scale_invariant_signal_distortion_ratio(mixture_audio, target_audio)

            metrics_dict["sdr"] += sdr.sum().cpu().item()
            metrics_dict["snr"] += snr.sum().cpu().item()
            metrics_dict["sisnr"] += sisnr.sum().cpu().item()
            metrics_dict["sisdr"] += sisdr.sum().cpu().item()
            metrics_dict["mixture_sdr"] += mixture_sdr.sum().cpu().item()
            metrics_dict["mixture_snr"] += mixture_snr.sum().cpu().item()
            metrics_dict["mixture_sisnr"] += mixture_sisnr.sum().cpu().item()
            metrics_dict["mixture_sisdr"] += mixture_sisdr.sum().cpu().item()
            n_samples += source_audio.size(0)

            rows.extend({"name": name, "snr": snr_val} for name, snr_val in zip(source_file, snr.cpu().tolist()))

        for key in list(metrics_dict.keys()):
            metrics_dict[key] /= max(n_samples, 1)
        metrics_dict["sdri"] = metrics_dict["sdr"] - metrics_dict["mixture_sdr"]
        metrics_dict["snri"] = metrics_dict["snr"] - metrics_dict["mixture_snr"]
        metrics_dict["sisnri"] = metrics_dict["sisnr"] - metrics_dict["mixture_sisnr"]
        metrics_dict["sisdri"] = metrics_dict["sisdr"] - metrics_dict["mixture_sisdr"]
        results[stem] = metrics_dict
        pd.DataFrame(rows).to_csv(out_dir / f"{stem}_snr.csv", index=False)
        (out_dir / f"{stem}_reference.json").write_text(json.dumps(metrics_dict, indent=2) + "\n")
    return results


def evaluate_pesq(pred_symlink_root: Path, gt_symlink_root: Path, out_dir: Path):
    dataset = SimpleDataset(str(pred_symlink_root / "speech"), str(gt_symlink_root / "speech"), False, "avdnr", "speech")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)
    total = 0.0
    num_files = 0
    for source_audio, target_audio, _ in tqdm.tqdm(data_loader, desc="PESQ speech"):
        pred_audios = source_audio.numpy()
        gt_audios = target_audio.numpy()
        batch_scores = pesq_batch(16000, gt_audios, pred_audios, "wb", n_processor=8)
        total += sum(batch_scores)
        num_files += pred_audios.shape[0]
    score = total / max(num_files, 1)
    (out_dir / "speech_pesq.txt").write_text(f"PESQ: {score}, num_files: {num_files}\n")
    return score


def aggregate_summary(model_name: str, distribution: dict, reference: dict, pesq_score: float):
    stem_rows = []
    for stem in STEMS:
        row = {
            "stem": stem,
            "frechet_audio_distance": distribution[stem].get("frechet_audio_distance"),
            "kullback_leibler_divergence_sigmoid": distribution[stem].get("kullback_leibler_divergence_sigmoid"),
            "sisdri": reference[stem].get("sisdri"),
        }
        stem_rows.append(row)

    summary = {
        "model_name": model_name,
        "stem_metrics": stem_rows,
        "aggregates": {
            "FAD": float(np.mean([row["frechet_audio_distance"] for row in stem_rows])),
            "KL": float(np.mean([row["kullback_leibler_divergence_sigmoid"] for row in stem_rows])),
            "SI-SDRi": float(np.mean([row["sisdri"] for row in stem_rows])),
            "PESQ": float(pesq_score),
            "WPR": None,
        },
        "notes": [
            "FAD, KL, and SI-SDRi are averaged across speech, sfx, and music.",
            "PESQ is evaluated on speech only.",
            "WPR is not recomputed here because its dedicated PANN-based pipeline is not packaged in cass_cvpr.",
        ],
    }
    return summary


def evaluate_single_stem(stem: str, pred_symlink_root: Path, gt_symlink_root: Path, results_out: Path, device_name: str):
    with threadpool_limits(limits=4):
        distribution = evaluate_distribution_metrics(
            pred_symlink_root,
            gt_symlink_root,
            results_out,
            stems=[stem],
            device_name=device_name,
        )
        reference = evaluate_reference_metrics(
            pred_symlink_root,
            gt_symlink_root,
            results_out,
            stems=[stem],
            device_name=device_name,
        )
    return distribution[stem], reference[stem]


def run_parallel_stem_workers(args, pred_symlink_root: Path, gt_symlink_root: Path, results_out: Path):
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        return None, None

    worker_plan = {
        "speech": ("0", "cuda:0"),
        "sfx": ("1", "cuda:0"),
        "music": ("0", "cuda:0"),
    }
    procs = []
    for stem in STEMS:
        visible_gpu, worker_device = worker_plan[stem]
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--pred-root", str(args.pred_root),
            "--gt-root", str(args.gt_root),
            "--symlink-root", str(args.symlink_root),
            "--results-out", str(args.results_out),
            "--model-name", args.model_name,
            "--stem", stem,
            "--device", worker_device,
        ]
        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = visible_gpu
        procs.append((stem, visible_gpu, subprocess.Popen(cmd, cwd=str(ROOT), env=worker_env)))

    failed = []
    for stem, visible_gpu, proc in procs:
        rc = proc.wait()
        if rc != 0:
            failed.append((stem, visible_gpu, rc))

    if failed:
        raise RuntimeError(f"Parallel stem workers failed: {failed}")

    distribution = {}
    reference = {}
    for stem in STEMS:
        distribution[stem] = json.loads((results_out / f"{stem}_distribution.json").read_text())
        reference[stem] = json.loads((results_out / f"{stem}_reference.json").read_text())
    return distribution, reference


def main():
    parser = argparse.ArgumentParser(description="Evaluate corrected AVDnR outputs and write machine-readable summaries.")
    parser.add_argument("--pred-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--symlink-root", required=True)
    parser.add_argument("--results-out", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--stem", choices=STEMS)
    parser.add_argument("--device")
    args = parser.parse_args()

    pred_root = Path(args.pred_root)
    gt_root = Path(args.gt_root)
    symlink_root = Path(args.symlink_root)
    results_out = Path(args.results_out)
    results_out.mkdir(parents=True, exist_ok=True)

    if args.stem:
        gt_symlink_root = symlink_root / "gt_avdnr_test"
        pred_symlink_root = symlink_root / "pred_avdnr"
        distribution, reference = evaluate_single_stem(
            args.stem,
            pred_symlink_root,
            gt_symlink_root,
            results_out,
            args.device or "cuda:0",
        )
        print(json.dumps({"stem": args.stem, "distribution": distribution, "reference": reference}, indent=2))
        return

    gt_symlink_root, pred_symlink_root = prepare_symlink_roots(gt_root, pred_root, symlink_root)
    distribution, reference = run_parallel_stem_workers(args, pred_symlink_root, gt_symlink_root, results_out)
    if distribution is None or reference is None:
        with threadpool_limits(limits=4):
            distribution = evaluate_distribution_metrics(pred_symlink_root, gt_symlink_root, results_out, device_name=args.device)
            reference = evaluate_reference_metrics(pred_symlink_root, gt_symlink_root, results_out, device_name=args.device)
    with threadpool_limits(limits=4):
        pesq_score = evaluate_pesq(pred_symlink_root, gt_symlink_root, results_out)
    summary = aggregate_summary(args.model_name, distribution, reference, pesq_score)
    (results_out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(summary["stem_metrics"]).to_csv(results_out / "stem_metrics.csv", index=False)
    pd.DataFrame([summary["aggregates"]]).to_csv(results_out / "aggregate_metrics.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
