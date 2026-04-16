
from math import sqrt
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import Tensor
import copy

""" Diffusion Classes """

import numpy as np


class  ReFlow(nn.Module):
    """
    Rectified Flow Model
    
    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        t_sampling_type: str = "uniform", # "uniform" or "logit_normal" or "near_data"
        infer_steps: int = 4,
        shift: float = 1,
        loss_reweight: bool = True,
    ):
        super().__init__()
        print(f"shift: {shift}")
        self.t_sampling_type = t_sampling_type
        self.infer_steps = infer_steps
        self.shift = shift
        
        # define cosine schedule
        self.n_timestep = n_timestep = 1000
        
        self.add_recon_reg = False
        self.loss_reweight = loss_reweight

    def forward(self, net: nn.Module, x: Tensor, **kwargs) -> Tensor:

        batch, device = x.shape[0], x.device
        
        # if x.ndim==3:
        #     sources = x[:, :3, ...]
        #     mixture = x[:, 3, ...].unsqueeze(1)
        #     t_shape = [batch, 1, 1]
        # else:
        #     sources = x[:, :6, ...]
        #     mixture = x[:, 6:, ...]
        
        t_shape = [batch, 1, 1, 1]

        # in order to make the model relay on mixture more, random zero out some sources
        zero_out = (torch.rand(batch, device=device) < 0.5).to(x.dtype)

        if self.t_sampling_type == "uniform":
            # Sample amount of noise to add for each batch element
            t = torch.rand(batch, device=device)
            t_scaled = t * self.n_timestep
            t_norm = t.view(*t_shape)
        elif self.t_sampling_type == "logit_normal":
            # logit normal sampling in SD3
            t = torch.randn(batch, device=device)
            t = 1.0 / (1.0 + torch.exp(-t))
            t_scaled = t * self.n_timestep
            t_norm = t.view(*t_shape)
        elif self.t_sampling_type == "near_data":
            # sample t extensively near 1 (= data)
            shape, scale = 1, 0.1
            t = np.random.gamma(shape, scale, batch)
            t = 1 - t
            t = np.clip(t, 0.0, 0.9999)
            t = torch.Tensor(t).to(device)
            # print(t.mean().item())
            t_scaled = t * self.n_timestep
            t_norm = t.view(*t_shape)
        else:
            raise ValueError(f"t_sampling_type must be 'uniform' or 'logit_normal', got {self.t_sampling_type}")

        if self.loss_reweight:
            assert self.t_sampling_type=="uniform"

            x1 = x
            x0 = torch.randn_like(x1)

            loss_dict = {}

            ut = x1 - x0
            t_unsqueeze = t_norm
            
            xt = t_unsqueeze * x0 + (1. - t_unsqueeze) * x1 
            # model_input = torch.cat([xt, mixture], dim=1)
            v_pred = net(xt, t_scaled, **kwargs)

            loss_simple = F.mse_loss(v_pred, ut, reduction="none")
            loss_v_pred = reduce(loss_simple, 'b ... -> b', 'mean')
            loss_dict.update({'loss_v_pred': loss_v_pred.mean()})

            t_cont = t_unsqueeze.squeeze().clamp(1e-5, 1. - 1e-5)
            lognorm_weights = torch.exp(-0.5 * torch.log(t_cont / ( 1 - t_cont)) ** 2) * 0.398942 / (t_cont * (1 - t_cont))
            # loss = torch.mean(lognorm_weights[:, None, None] * loss_simple)
            loss = torch.mean(lognorm_weights * loss_v_pred)
            loss_dict.update({'loss': loss})
            
            return loss_dict

        else:
            sources = x
            # Add noise to input
            noise = torch.randn_like(sources)
            # print(sources.shape)
            zt = t_norm * noise + (1 - t_norm) * sources 

            # flow prediction target
            target = sources - noise # should be (noise - sources), but we can keep this for now to avoid introduce more errors
            
            # model prediction
            v_pred = net(zt, t_scaled, **kwargs)
            
            # compute loss
            loss_v_pred = F.mse_loss(v_pred, target.to(torch.float32), reduction="none")
            loss_v_pred = reduce(loss_v_pred, 'b ... -> b', 'mean')
            loss_v_pred = loss_v_pred.mean()
            # mixtures_recon_loss = F.mse_loss(mixture_pred, mixture.squeeze(1), reduction="mean")
            loss = loss_v_pred
            
            loss_dict = {
                "loss_v_pred": loss_v_pred,
            }
            
            
            if self.add_recon_reg:
                pred_sources = v_pred + noise
                loss_recon = F.l1_loss(pred_sources, sources, reduction="none").mean(dim=(1,2))
                loss_recon = (1 - t_norm)**2 * loss_recon
                loss_recon = loss_recon.mean()
            
                loss = loss + loss_recon * 0.1
                loss_dict["loss_recon"] = loss_recon

            loss_dict["loss"] = loss
            return loss_dict

    @torch.inference_mode()
    def sample(self, net: nn.Module, noise: Tensor, **model_kwargs):

        # vid_cond = {}
        # old sampling code
        shift = self.shift
            
        # old sampling code
        # print("shift =", shift)
        batch, device = noise.shape[0], noise.device
        z_t = noise
        step_max = 1.
        step_min = 0. # sd3 0.002994012087583542
        euler_steps = np.linspace(step_max, step_min, self.infer_steps)
        euler_steps = shift * euler_steps / ( 1 + (shift - 1) * euler_steps)
        euler_steps = (euler_steps * self.n_timestep).tolist()
        eular_step_pair = list(zip(euler_steps[:-1], euler_steps[1:]))
        # z_ts = [z_t]
        
        for i, (step, step_next) in enumerate(eular_step_pair):
            ts = torch.ones(batch,device=device) * step
            pred_v = net(z_t, ts, **model_kwargs)
            step_size = (step - step_next) / self.n_timestep
            z_t = z_t + pred_v * step_size
            # z_ts.append(z_t)
        return z_t

    # @torch.inference_mode()
    # def sample(self, net: nn.Module, noise: Tensor, **model_kwargs):

    #     # vid_cond = {}
    #     # old sampling code
    #     shift = self.shift
            
    #     # old sampling code
    #     # print("shift =", shift)
    #     batch, device = noise.shape[0], noise.device
    #     z_t = noise
    #     step_max = 1.
    #     step_min = 0. # sd3 0.002994012087583542
    #     euler_steps = np.linspace(step_max, step_min, self.infer_steps)
    #     euler_steps = shift * euler_steps / ( 1 + (shift - 1) * euler_steps)
    #     euler_steps = (euler_steps * self.n_timestep).tolist()
    #     eular_step_pair = list(zip(euler_steps[:-1], euler_steps[1:]))
    #     # z_ts = [z_t]
        
    #     mixture = model_kwargs['mixture_latents']

    #     zero_mix_kwargs = copy.deepcopy(model_kwargs)
    #     zero_mix_kwargs['mixture_latents'] = torch.zeros_like(model_kwargs['mixture_latents'])

    #     for i, (step, step_next) in enumerate(eular_step_pair):
            
    #         if i < len(eular_step_pair) // 3 * 2:
    #             model_kwargs_input = zero_mix_kwargs
    #         else:
    #             model_kwargs_input = model_kwargs

    #         ts = torch.ones(batch,device=device) * step
    #         pred_v = net(z_t, ts, **model_kwargs_input)
    #         step_size = (step - step_next) / self.n_timestep
    #         z_t = z_t + pred_v * step_size
    #         # z_ts.append(z_t)
    #     return z_t

