# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.amp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from accelerate import Accelerator

from models_avdnr import SiT_models
from transport.RFM import ReFlow
from train_utils import parse_transport_args
import wandb_utils

from data.data_AVDnR import MultiSourceDataset
from einops import rearrange
from spec_utils import audio2spec, spec2audio

import torchaudio


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir, rank=0):
    """
    Create a logger that writes to a log file and stdout.
    """
    if rank == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger




#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new SiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    accelerator = Accelerator()
    device = accelerator.device
    print(device)
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    local_batch_size = int(args.global_batch_size // world_size)

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_name = f"{experiment_index:03d}-{model_string_name}-" \
                        f"{args.path_type}-{args.prediction}-{args.loss_weight}-{args.exp_name}"
        experiment_dir = f"{args.results_dir}/{experiment_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, rank)
        logger.info(f"Experiment directory created at {experiment_dir}")


        if args.wandb:
            entity = os.environ["ENTITY"]
            project = os.environ["PROJECT"]
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None, rank)

    model = SiT_models[args.model](in_channels=8, out_channels=6, attention_head_dim=args.attention_head_dim).to(device)


    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(state_dict["model"])
        logger.info(f"Loaded model from {ckpt_path}")

    ema = deepcopy(model).to(device)

    requires_grad(ema, False)
    
    transport = ReFlow()
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    mixture_name = "mix" if "dnr_v2" in args.audio_files_dir else "mixture"
    dataset = MultiSourceDataset(
        sr=16000,
        sample_length=130816,
        audio_files_dir=args.audio_files_dir,
        stems=['speech', 'sfx', 'music'],
        mixture_name=mixture_name,
        audio_feat_type="waveform",
        visual_encoder_type=None,
        enlarge_dataset=100,
        limit_samples=-1,
        eval_mode=False,
        load_whole=False,
        vid_stem_type=None,
        vid_stem_mode=None,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} ")


    val_batch_size = 10
    world_size = world_size
    dataset_val = MultiSourceDataset(
        sr=16000,
        sample_length=130816,
        audio_files_dir=args.audio_files_dir,
        stems=['speech', 'sfx', 'music'],
        mixture_name=mixture_name,
        audio_feat_type="waveform",
        visual_encoder_type=None,
        enlarge_dataset=1,
        limit_samples=val_batch_size * world_size,
        eval_mode=True,
        load_whole=False,
        vid_stem_type=None,
        vid_stem_mode=None,
        rank=rank,
        world_size=world_size,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()
    model, opt = accelerator.prepare(model, opt)

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        for batch in loader:
            opt.zero_grad()
            waveforms = batch.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                wav_spec = audio2spec(waveforms)

                B, N, C, H, W = wav_spec.shape
                waveforms_latents = wav_spec
                mixture_latents = waveforms_latents[:, -1, ...].reshape(B, C, H, W).contiguous()
                sources_latents = waveforms_latents[:, :-1, ...].reshape(B, 3*C, H, W).contiguous()

                model_kwargs = dict(mixture_latents=mixture_latents)

                loss_dict = transport(model, sources_latents, **model_kwargs)
                loss = loss_dict["loss"].mean()

            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model.module)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log(
                        { "train loss": avg_loss, "train steps/sec": steps_per_sec },
                        step=train_steps
                    )
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0  or train_steps == 1:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    if train_steps == 1:
                        os.remove(f"{checkpoint_dir}/{train_steps:07d}.pt")

                dist.barrier()

            if 0 < args.max_steps <= train_steps:
                break
            
            if train_steps % args.sample_every == 0 or train_steps == 1:
                logger.info("Generating EMA samples...")

                for batch in loader_val:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        waveforms, mixture, dirpaths = batch
                        mixture_latents = audio2spec(mixture.unsqueeze(1).to(device))
                        B, _, C, H, W = mixture_latents.shape
                        mixture_latents = mixture_latents.reshape(B, C, H, W).contiguous()

                        z = torch.randn(B, 3*C, H, W, device=device)

                        model_kwargs = dict(mixture_latents=mixture_latents, cfg_scale=args.cfg_scale)
                        model_fn = ema.forward_with_cfg

                        with torch.inference_mode():
                            samples = transport.sample(model_fn, z, **model_kwargs)

                dist.barrier()

                if train_steps == 1:
                    waveforms = waveforms.to(device)
                    gt_waveforms_spec = audio2spec(waveforms)
                    gt_waveforms_recon = spec2audio(gt_waveforms_spec)
                    samples_ = spec2audio(samples)
                    
                    assert samples_.shape == gt_waveforms_recon.shape, f"Samples shape {samples.shape} != gt_waveforms_spec shape {gt_waveforms_spec.shape}"
                    samples = gt_waveforms_spec

                samples = spec2audio(samples.reshape(B, 3, C, H, W).contiguous())
                out_samples = torch.zeros((val_batch_size*world_size, 3, mixture.size(-1)), device=device)
                dist.all_gather_into_tensor(out_samples, samples)

                if rank == 0:
                    sample_folder_dir = f"{experiment_dir}/samples"
                    os.makedirs(sample_folder_dir, exist_ok=True)
                    for j, dirpath in enumerate(out_samples):
                        eval_save_dir = f"{sample_folder_dir}/{j}"
                        os.makedirs(eval_save_dir, exist_ok=True)
                        
                        for i, stem in enumerate(dataset.stems):
                            torchaudio.save(f"{eval_save_dir}/{stem}-{train_steps}.wav", out_samples[j][i].unsqueeze(0).cpu(), sample_rate=16000)

                logging.info("Generating EMA samples done.")

        if 0 < args.max_steps <= train_steps:
            break

    model.eval()

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--audio_files_dir", type=str, default="")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="UNet2d_S2")
    parser.add_argument("--attention_head_dim", type=int, default=64, choices=[8, 64], help="set 64 to enable flash attention")
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--cfg-scale", type=float, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom SiT checkpoint")
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop training after this many steps (0 = no limit).")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
