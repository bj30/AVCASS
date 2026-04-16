# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from einops import rearrange, repeat
from diffusers import DDPMScheduler, UNet2DModel, UNet2DConditionModel

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert (dim % 2) == 0, 'dimension must be divisible by 2'
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = pos - seq_start_pos[..., None]

        emb = torch.einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale

class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        patch_size=2,
        in_channels=128,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        # self.t_embedder = TimestepEmbedder(hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        # num_patches = self.x_embedder.num_patches
        # # Will use fixed sin-cos embedding:
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)


        self.project_in_mixture = nn.Linear(in_channels*3, hidden_size, bias=True)
        self.project_in_xt = nn.Linear(in_channels*3, hidden_size, bias=True)
        self.pos_emb = ScaledSinusoidalEmbedding(hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, 1, self.out_channels*3)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w = self.x_embedder.proj.weight.data
        # nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def forward(self, x, t, mixture_latents):
        """
        Forward pass of SiT.
        x: B, N, C, T
        t: (N,) tensor of diffusion timesteps
        mixture_latents: B, C, T
        """
        B, N, C, T = x.shape
        mixture = mixture_latents.permute(0, 2, 1).repeat(1,1,3) # B, T, C*3
        mixture = self.project_in_mixture(mixture) # B, T, D

        xt = x.permute(0, 1, 3, 2) # B, N, T, C
        xt = rearrange(xt, 'b n t c -> b t (n c)') # B, T, N*C
        xt = self.project_in_xt(xt) # B, T, D

        x = torch.cat([xt, mixture], dim=1) # B, 2*T, D
        x = x + self.pos_emb(x)
        t = self.t_embedder(t)                  
        c = t

        for block in self.blocks:
            x = block(x, c)                      
        x = x[:, :T, :] # B, T, D
        x = self.final_layer(x, c)  # B, T, N*C
        x = rearrange(x, 'b t (n c) -> b n c t', n=N, c=self.out_channels) # B, N, C, T

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of SiT, but also batches the unconSiTional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)



class UNet2d(nn.Module):
    def __init__(
        self,
        in_channels: int = 4, # 4
        out_channels = 3,
        block_out_channels = (128, 128, 256, 512), #(224, 448, 672, 896),
        attention_head_dim = 8,
        dropout_prob = 0.1,
        visual_encoder_type = None,
        freeze_image_model = True,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        
        self.dropout_prob = dropout_prob
        self.visual_encoder_type = visual_encoder_type
        
        
        if self.visual_encoder_type is None:
            self.model = UNet2DModel(
                sample_size=(128,256),
                in_channels=in_channels,
                out_channels=out_channels,
                attention_head_dim=attention_head_dim,
                block_out_channels=block_out_channels,
            )
        else:
            self.freeze_image_model = freeze_image_model
            self.image_model, visual_feat_dim = self.init_visual_encoder(visual_encoder_type)
            self.model = UNet2DConditionModel(
                            sample_size=(128,256),
                            in_channels=in_channels,
                            out_channels=out_channels,
                            encoder_hid_dim=visual_feat_dim,
                            encoder_hid_dim_type="text_proj",
                            cross_attention_dim=visual_feat_dim,
                            attention_head_dim=16,
                            block_out_channels=block_out_channels,
                        )
        # if attention_head_dim == 64:
        from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
        self.model.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)


    def init_visual_encoder(self, visual_encoder_type):
        if visual_encoder_type == 'cavp':
            from visual_encoders.cavp.demo_util import instantiate_from_config, init_from_ckpt
            from omegaconf import OmegaConf
            image_feat_dim = 512
            out_dim = 64
            aligner_type = "image_linear"
            # cavp_config_path = "./config/Stage1_CAVP.yaml"
            config = OmegaConf.create({"model": {"target":"visual_encoders.cavp.model.cavp_model.CAVP_Inference_VideoOnly", 
                                    "params":{"video_encode": "Slowonly_pool", "spec_encode": "cnn14_pool", 
                                    "embed_dim": image_feat_dim, "video_pretrained": True, "audio_pretrained": True}}})
            image_model = instantiate_from_config(config.model) #.to(device)
            # Loading Model from:
            ckpt_path = "/mnt/bear1/users/zhangkang/MSDM/video_encoder_st.pt"
            print("Loading CAVP Model from: {}".format(ckpt_path))
            image_model = init_from_ckpt(ckpt_path, image_model)
            
        elif visual_encoder_type == 'talknet':
            from visual_encoders.visual_encoder import visualFrontend
            image_feat_dim = 64
            out_dim = 64
            aligner_type = None
            image_model = visualFrontend(image_feat_dim)
            image_model.init_from_ckpt()
            
        elif visual_encoder_type == 'SSLAlignment':
            from visual_encoders.SSL_vid_encoder import AVENetVidEncoder
            image_model = AVENetVidEncoder()
            image_feat_dim = 512
            out_dim = 512
            aligner_type = "ssl_video"
            
        elif visual_encoder_type == 'both':
            from visual_encoders.both import EnsembleEncoder
            image_model = EnsembleEncoder()
            image_feat_dim = 512
            out_dim = 256
            aligner_type = "ssl_video"
            
        else:
            raise ValueError(f"Visual encoder {visual_encoder_type} is not implemented")
        
        # disable the gradients of the visual encoder
        if self.freeze_image_model:
            image_model.eval()
            for param in image_model.parameters():
                param.requires_grad = False

        if visual_encoder_type == "both":
            out_dim = 640 # 64*6+256
        return image_model, image_feat_dim

    def forward(self, x, t, mixture_latents, vid=None):
        if mixture_latents.ndim == 3:
            mixture_latents = mixture_latents.unsqueeze(1)
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        #     x = torch.cat([x, mixture_latents], dim=1)
        #     output = self.model(x, timestep=t)
        #     x = output['sample']
        if self.training and self.dropout_prob > 0:
            # randomly drop mixture_latents
            drop_ids = torch.rand(mixture_latents.shape[0], device=mixture_latents.device) < self.dropout_prob
            # mixture_latents: B, 1, C, T
            mixture_latents = torch.where(drop_ids[:, None, None, None], torch.zeros_like(mixture_latents), mixture_latents)

        x = torch.cat([x, mixture_latents], dim=1)
        if self.visual_encoder_type is None:
            output = self.model(x, timestep=t)
        else:
            vid_feature = self.forward_video(vid) # (B, T=32, C=512)
            # print(x.shape, t.shape, vid_feature.shape)
            
            output = self.model(x, timestep=t, encoder_hidden_states=vid_feature)
        x = output['sample']
        return x
    
    def forward_video(self, inputBatch):
        # check the input value range is in [0 255]
        # assert inputBatch.max() > 1 and inputBatch.max() <= 255
        if self.visual_encoder_type == 'both':
            # print(inputBatch['speech'].shape, inputBatch['sfx'].shape)
            speech_feats_ori, sfx_feats_ori = self.image_model(inputBatch)
            # aligned_feats = self.image_aligner(sfx_feats_ori)# (B, T=32, C=out_dim=256)
            aligned_feats = sfx_feats_ori
            
            sp_feat_split = []
            batch_size = speech_feats_ori.shape[0]
            feat_fps = 4
            sp_feat_fps = 25
            agg_feat_len = sp_feat_fps//feat_fps
            for t in range(speech_feats_ori.shape[1]//sp_feat_fps):
                for i in range(feat_fps):
                    sp_feat_split.append(speech_feats_ori[:, t*sp_feat_fps+i*agg_feat_len:t*sp_feat_fps+(i+1)*agg_feat_len].reshape(batch_size, -1)) # [B, 6, 64] -> [B, 384]
                # sp_feat_split.append(speech_feats_ori[:, t*sp_feat_fps+6:t*sp_feat_fps+12].view(batch_size, -1))
                # sp_feat_split.append(speech_feats_ori[:, t*sp_feat_fps+12:t*sp_feat_fps+18].view(batch_size, -1))
                # sp_feat_split.append(speech_feats_ori[:, t*sp_feat_fps+18:t*sp_feat_fps+24].view(batch_size, -1))
            speech_feats_splitted = torch.stack(sp_feat_split, dim=1) # [B, 32, 384]
            # print(speech_feats_splitted.shape, aligned_feats.shape)
            concat_feat = torch.cat([speech_feats_splitted, aligned_feats], dim=2) # [B, 32, 640]
            return concat_feat
        else:
            inputBatch = inputBatch / 255.0 # normalize to range [0 1]
        
        if self.visual_encoder_type == 'cavp':
            # batch: [B, T, C, H, W]
            # inputBatch.shape = (B, T, C=3, W=224, H=224)
            if inputBatch.ndim==4:
                inputBatch.unsqueeze(0)
                
            image_feats_ori = self.image_model.encode_video(inputBatch, normalize=True, pool=False) # (B, T, C=feat_dim)
            return image_feats_ori
            # if self.image_aligner is None:
            #     return image_feats_ori
            # else:
            #     # return image_feats
            #     image_feats = rearrange(image_feats_ori, 'b t c -> b c t 1').contiguous()
            #     aligned_image_feats = self.image_aligner(image_feats, None)
            #     # aligned_image_feats = self.prep_feats(aligned_image_feats, is_audio=False)
            #     aligned_image_feats = rearrange(aligned_image_feats.squeeze(-1), 'b c t -> b t c').contiguous()
            #     return aligned_image_feats

                    
        elif self.visual_encoder_type == 'talknet':
            # batch: [B, T, H=112, W=112]
            if inputBatch.ndim == 3:
                inputBatch.unsqueeze(0)
            
            # B,T,C,W,H = inputBatch.shape
            # inputBatch = rearrange(inputBatch, 'b t c h w -> (b t) c h w')
            # # inputBatch = self.img_normalize(inputBatch) # normalize using ImageNet statistics
            # inputBatch = rearrange(inputBatch, '(b t) c h w -> b t c h w', b=B)
            image_feats_ori = self.image_model.encode_video(inputBatch, normalize=True, pool=False) # (B, T, C=feat_dim)
            return image_feats_ori
            # if self.image_aligner is None:
            #     return image_feats_ori
            # else:
            #     # return image_feats
            #     image_feats = rearrange(image_feats_ori, 'b t c -> b c t 1').contiguous()
            #     aligned_image_feats = self.image_aligner(image_feats, None)
            #     # aligned_image_feats = self.prep_feats(aligned_image_feats, is_audio=False)
            #     aligned_image_feats = rearrange(aligned_image_feats.squeeze(-1), 'b c t -> b t c').contiguous()
            #     return aligned_image_feats
            
        elif self.visual_encoder_type == 'SSLAlignment':
            # batch: [B, T, C, H, W]
            # inputBatch.shape = (B, T, C=3, W=224, H=224)
            self.image_model.eval()
            with torch.no_grad():
                image_feats_ori = self.image_model(inputBatch) # (B, T, C=feat_dim, W=14, H=14)
            # aligned_feats = self.image_aligner(image_feats_ori) # (B, T, C=out_dim)
            aligned_feats = image_feats_ori.mean(-1).mean(-1)
            return aligned_feats


    def forward_with_cfg(self, x, t, mixture_latents, vid=None, cfg_scale=0.0):
        """
        Forward pass of SiT, but also batches the unconSiTional forward pass for classifier-free guidance.
        """
        if cfg_scale == 0:
            return self.forward(x, t, mixture_latents, vid=vid)
        else:
            assert cfg_scale >= 1.0
            cond_eps = self.forward(x, t, mixture_latents, vid=vid)
            uncond_eps = self.forward(x, t, torch.zeros_like(mixture_latents), vid=vid)
            output = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            return output

    # def forward(
    #     self,
    #     x,
    #     time = None,
    #     *,
    #     features = None, # visual conditioning
    #     channels_list = None,
    #     embedding = None,
    #     return_attn = False,
    # ) :
    #     # sample: (B, C, H, W)
    #     # timestep: float or int
    #     # encoder_hidden_states: (B, L, feature_dim)
    #     if features is not None:
    #         output = self.model(sample=x, timestep=time, encoder_hidden_states=features)
    #     else:
    #         output = self.model(sample=x, timestep=time)
    #     x = output['sample']
    #     return x



#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def UNet2d_Small(**kwargs):
    """
    16m parameters
    """
    return UNet2d(block_out_channels = (64, 128, 128, 224), **kwargs)

def UNet2d_S2(**kwargs):
    """
    30m parameters
    """
    return UNet2d(block_out_channels = (128, 128, 256, 256), **kwargs)

def UNet2d_S3(**kwargs):
    """
    120m parameters
    """
    return UNet2d(block_out_channels = (256, 256, 512, 512), **kwargs)

SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
    'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
    'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8, 
    'UNet2d': UNet2d,      'UNet2d_Small': UNet2d_Small,
    'UNet2d_S2': UNet2d_S2, 'UNet2d_S3': UNet2d_S3
}
