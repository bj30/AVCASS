# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from models_avdnr_zero_conv_2vid import SiT_models
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
import sys
from glob import glob
import pandas as pd
import torch.nn as nn

import torchaudio
from torch.utils.data import DataLoader
from data.data_AVDnR import MultiSourceDataset
from spec_utils import audio2spec, spec2audio
from transport.RFM import ReFlow
from torchmetrics.functional.audio import signal_distortion_ratio, signal_noise_ratio, scale_invariant_signal_noise_ratio, scale_invariant_signal_distortion_ratio
from visual_backbones import init_visual_encoder, forward_video


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(mode, args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    image_model, image_feat_dim = init_visual_encoder(args.visual_encoder_type)
    image_model = image_model.to(device)
    
    forward_video_fn = torch.compile(forward_video)
    
    # Load model:
    model_init_kwargs = dict(in_channels=8, out_channels=6, attention_head_dim=args.attention_head_dim, visual_feat_dim=image_feat_dim)
    model = SiT_models[args.model](**model_init_kwargs).to(device)
    ckpt_path = args.ckpt or f"SiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = torch.load(ckpt_path, weights_only=False, map_location="cpu")['ema']
    model.load_state_dict(state_dict)
    model.eval()
    
    
    transport = ReFlow(
        infer_steps=args.num_sampling_steps,
    )

    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    if args.folder_name:
        folder_name = args.folder_name
        exclude_list = []
        if '/AVDnR' in args.audio_files_dir:
            audio_directory = os.path.join(args.audio_files_dir, 'test') if os.path.exists(os.path.join(args.audio_files_dir, 'test')) else args.audio_files_dir
            if os.path.isdir(f"{args.sample_dir}/{folder_name}"):
                for case_name in os.listdir(f"{args.sample_dir}/{folder_name}"):
                    case_dir = os.path.join(args.sample_dir, folder_name, case_name)
                    if not os.path.isdir(case_dir):
                        continue
                    for file_name in os.listdir(case_dir):
                        file_dir = os.path.join(case_dir, file_name)
                        if not os.path.isdir(file_dir):
                            continue
                        exists = 0
                        for stem in ["speech", "sfx", "music"]:
                            if os.path.exists(os.path.join(file_dir, f"{stem}.wav")):
                                exists += 1
                        if exists == 3:
                            exclude_list.append(f"{audio_directory}/{case_name}/{file_name}")

        elif '/dnr_v2_16k' in args.audio_files_dir:
            audio_directory = '/mnt/lynx1/datasets/dnr_v2_16k/tt'
            if os.path.isdir(f"{args.sample_dir}/{folder_name}"):
                for case_name in os.listdir(f"{args.sample_dir}/{folder_name}"):
                        exists =0 
                        for stem in ["speech", "sfx", "music"]:
                            if os.path.exists(os.path.join(args.sample_dir, folder_name, case_name, f"{stem}.wav")):
                                exists += 1
                        if exists == 3:
                            exclude_list.append(f"{audio_directory}/{case_name}/")
            
        else:
            os.makedirs(f"{args.sample_dir}/{folder_name}", exist_ok=True)
        os.makedirs(f"{args.sample_dir}/{folder_name}", exist_ok=True)
        sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    else:
        model_string_name = args.model.replace("/", "-")
        ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
        if mode == "ODE":
            folder_name = f"{model_string_name}-{ckpt_string_name}-" \
                    f"cfg-{args.cfg_scale}-{args.per_proc_batch_size}-"\
                    f"{mode}-{args.num_sampling_steps}-{args.sampling_method}"
        elif mode == "SDE":
            folder_name = f"{model_string_name}-{ckpt_string_name}-" \
                        f"cfg-{args.cfg_scale}-{args.per_proc_batch_size}-"\
                        f"{mode}-{args.num_sampling_steps}-{args.sampling_method}-"\
                        f"{args.diffusion_form}-{args.last_step}-{args.last_step_size}"
        experiment_index = len(glob(f"{args.sample_dir}/*"))
        experiment_index = torch.tensor(experiment_index).to(device)
        dist.broadcast(experiment_index, 0)
        experiment_index = experiment_index.item()
        sample_folder_dir = f"{args.sample_dir}/{experiment_index:03d}-{folder_name}"
        os.makedirs(sample_folder_dir, exist_ok=True)
        exclude_list = []

    if rank == 0:
        print(f"Saving .png samples at {sample_folder_dir}")
        

    vid_stem_type = ['speech'] if args.visual_encoder_type=='talknet' else ['sfx']
    mixture_name = "mix" if "dnr_v2" in args.audio_files_dir else "mixture"
    dataset = MultiSourceDataset(
        sr=16000,
        sample_length=130816,
        audio_files_dir=args.audio_files_dir,
        stems=['speech', 'sfx', 'music'],
        mixture_name=mixture_name,
        audio_feat_type="waveform",
        visual_encoder_type=args.visual_encoder_type,
        enlarge_dataset=1,
        limit_samples=-1,
        eval_mode=True,
        load_whole=True if args.load_whole == 1 else False,
        vid_stem_type=vid_stem_type,
        vid_stem_mode=None,
        rank=rank,
        world_size=dist.get_world_size(),
        exclude_list=exclude_list,
        root_dnrv3_dataset_path=args.root_dnrv3_dataset_path,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.per_proc_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    

    dist.barrier()

    pbar = tqdm(loader, desc=f"Sampling", disable=rank != 0)
    for idx, batch in enumerate(pbar):
        waveforms, mixture, vid, dirpaths = batch
        B, L = mixture.shape
        audio_length = L
        mixture = mixture.to(device)
        vid = vid.to(device)

        inference_audio_length = 130816
        assert audio_length >= inference_audio_length
        hop_length = int(inference_audio_length // 4 * 3)
        hop_length = int(inference_audio_length - 256 )

        overlap_count = torch.zeros((B, audio_length)).to(mixture)
        pred_audio = torch.zeros((B, 3, audio_length)).to(mixture)

        mixture_spec_chunked_list = []
        vid_feature_chunked_list = []
        index_list = []
        for i in range(0, audio_length, hop_length):
            if i+inference_audio_length >= audio_length:
                start = audio_length-inference_audio_length
                end = audio_length
            else:
                start = i
                end = i+inference_audio_length
            
            index_list.append([start, end])

            mixture_chunk = mixture[:, start:end]
            mixture_latents = audio2spec(mixture_chunk.unsqueeze(1).to(device))
            _, _, C, H, W = mixture_latents.shape
            mixture_latents = mixture_latents.reshape(B, C, H, W).contiguous()
            mixture_spec_chunked_list.append(mixture_latents)

            overlap_count[:, start:end] = overlap_count[:, start:end] + 1

            if args.visual_encoder_type=='cavp':
                load_fps = 4
            elif args.visual_encoder_type == 'SSLAlignment':
                load_fps = 4
            else:
                load_fps = 25
            sliding_vid_start = int(start / 16000 * load_fps)
            sliding_vid_end = sliding_vid_start + int(inference_audio_length / 16000 * load_fps)
            chunked_vid = vid[:, sliding_vid_start:sliding_vid_end, ...]
            with torch.autocast('cuda', dtype=torch.float16):
                vid_feature = forward_video_fn(image_model, chunked_vid, args.visual_encoder_type)
            vid_feature_chunked_list.append(vid_feature)
    

        model_fn = model.forward_with_cfg

        chunk_p_bar = tqdm(total=len(mixture_spec_chunked_list), desc=f"Chunk Sampling", disable=rank != 0)
        for (start, end), mixture_latents, vid_feature in zip(index_list, mixture_spec_chunked_list, vid_feature_chunked_list):
            noise = torch.randn(B, 3*C, H, W, device=device)
            vid_feature = vid_feature.half()
            model_kwargs = dict(mixture_latents=mixture_latents, cfg_scale=args.cfg_scale, vid=vid_feature)
            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.float16):
                    samples = transport.sample(model_fn, noise, **model_kwargs)
            samples = spec2audio(samples)
            pred_audio[:, :, start:end] = pred_audio[:, :, start:end] + samples
            chunk_p_bar.update(1)
        chunk_p_bar.close()

        pred_audio = torch.divide(pred_audio, overlap_count.unsqueeze(1)).clamp(-1, 1)
        samples = pred_audio

        waveforms = waveforms.to(device)
        mixture = mixture.to(device)

        for j, dirpath in enumerate(dirpaths):
            case_num, file_num = dirpath.split('/')[-2:]
            eval_save_dir = f"{sample_folder_dir}/{case_num}/{file_num}"
            os.makedirs(eval_save_dir, exist_ok=True)
            
            metrics_dict = {}
            for i, stem in enumerate(dataset.stems):
                torchaudio.save(f"{eval_save_dir}/{stem}.wav", samples[j][i].unsqueeze(0).cpu(), sample_rate=16000)

                metrics_dict[f"{stem}_sdr"] = signal_distortion_ratio(samples[j][i], waveforms[j][i]).cpu().item()
                metrics_dict[f"{stem}_snr"] = signal_noise_ratio(samples[j][i], waveforms[j][i]).cpu().item()
                metrics_dict[f"{stem}_sisnr"] = scale_invariant_signal_noise_ratio(samples[j][i], waveforms[j][i]).cpu().item()
                metrics_dict[f"{stem}_sisdr"] = scale_invariant_signal_distortion_ratio(samples[j][i], waveforms[j][i]).cpu().item()

                metrics_dict[f"{stem}_sisdri"] = metrics_dict[f"{stem}_sisdr"] - scale_invariant_signal_distortion_ratio(mixture[j], waveforms[j][i]).cpu().item()
                metrics_dict[f"{stem}_sisnri"] = metrics_dict[f"{stem}_sisnr"] - scale_invariant_signal_noise_ratio(mixture[j], waveforms[j][i]).cpu().item()
            
            df = pd.DataFrame(metrics_dict, index=[0])
            df.to_csv(f"{eval_save_dir}/metrics.csv", index=False)

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        results = []

        for parts in os.listdir(sample_folder_dir):
            parts_dir = os.path.join(sample_folder_dir, parts)
            if not os.path.isdir(parts_dir):
                continue
            for case in os.listdir(parts_dir):
                case_dir = os.path.join(parts_dir, case)
                if not os.path.isdir(case_dir):
                    continue
                for file in os.listdir(case_dir):
                    if file.endswith(".csv"):
                        df = pd.read_csv(os.path.join(case_dir, file))
                        results.append(df.to_numpy())

        results = np.concatenate(results, axis=0)
        avg_results = np.mean(results, axis=0)

        column_names = df.columns.to_list()
        df_avg_dict = {k: v for k, v in zip(column_names, avg_results)}
        df_avg = pd.DataFrame(df_avg_dict, index=[0])
        df_avg.to_csv(os.path.join(sample_folder_dir, "average_results.csv"), index=False)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    mode = "ODE"
    
    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"

    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="UNet2d_S2")
    parser.add_argument("--audio_files_dir", type=str, default="")
    parser.add_argument("--folder_name", type=str, default="", help="Optional folder name for eval unfinished inference.")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--visual_encoder_type", type=str, choices=["cavp", "talknet"], default="cavp")
    parser.add_argument("--attention_head_dim", type=int, default=64, choices=[8, 64], help="set 64 to enable flash attention")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=4)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=0.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--load_whole", type=int, choices=[0, 1], default=1)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a SiT checkpoint (default: auto-download a pre-trained SiT-XL/2 model).")
    parser.add_argument("--root_dnrv3_dataset_path", type=str, default="")
    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
    elif mode == "SDE":
        parse_sde_args(parser)

    args = parser.parse_args()
    main(mode, args)
