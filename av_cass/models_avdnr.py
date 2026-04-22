# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
from diffusers import UNet2DModel


class UNet2d(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels = 3,
        block_out_channels = (128, 128, 256, 512),
        attention_head_dim = 8,
        dropout_prob = 0.1,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.dropout_prob = dropout_prob

        self.model = UNet2DModel(
            sample_size=(128,256),
            in_channels=in_channels,
            out_channels=out_channels,
            attention_head_dim=attention_head_dim,
            block_out_channels=block_out_channels,
        )
        from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
        self.model.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)

    def forward(self, x, t, mixture_latents):
        if mixture_latents.ndim == 3:
            mixture_latents = mixture_latents.unsqueeze(1)

        if self.training and self.dropout_prob > 0:
            drop_ids = torch.rand(mixture_latents.shape[0], device=mixture_latents.device) < self.dropout_prob
            mixture_latents = torch.where(drop_ids[:, None, None, None], torch.zeros_like(mixture_latents), mixture_latents)

        x = torch.cat([x, mixture_latents], dim=1)
        output = self.model(x, timestep=t)
        return output['sample']

    def forward_with_cfg(self, x, t, mixture_latents, cfg_scale=0.0):
        if cfg_scale == 0:
            return self.forward(x, t, mixture_latents)
        else:
            assert cfg_scale >= 1.0
            cond_eps = self.forward(x, t, mixture_latents)
            uncond_eps = self.forward(x, t, torch.zeros_like(mixture_latents))
            return uncond_eps + cfg_scale * (cond_eps - uncond_eps)


def UNet2d_Small(**kwargs):
    """16m parameters"""
    return UNet2d(block_out_channels=(64, 128, 128, 224), **kwargs)

def UNet2d_S2(**kwargs):
    """30m parameters"""
    return UNet2d(block_out_channels=(128, 128, 256, 256), **kwargs)

def UNet2d_S3(**kwargs):
    """120m parameters"""
    return UNet2d(block_out_channels=(256, 256, 512, 512), **kwargs)

SiT_models = {
    'UNet2d': UNet2d,
    'UNet2d_Small': UNet2d_Small,
    'UNet2d_S2': UNet2d_S2,
    'UNet2d_S3': UNet2d_S3,
}
