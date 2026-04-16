import os

import torch


def _require_path(path, env_name):
    if not path:
        raise ValueError(f"Set {env_name} before running AV training or AV inference.")
    return path


def init_cavp(cavp_ckpt=None):
    from omegaconf import OmegaConf

    from visual_encoders.cavp.demo_util import init_from_ckpt, instantiate_from_config

    cavp_ckpt = _require_path(cavp_ckpt or os.environ.get("CAVP_CKPT"), "CAVP_CKPT")
    image_feat_dim = 512
    config = OmegaConf.create(
        {
            "model": {
                "target": "visual_encoders.cavp.model.cavp_model.CAVP_Inference_VideoOnly",
                "params": {
                    "video_encode": "Slowonly_pool",
                    "spec_encode": "cnn14_pool",
                    "embed_dim": image_feat_dim,
                    "video_pretrained": True,
                    "audio_pretrained": True,
                },
            }
        }
    )
    image_model = instantiate_from_config(config.model)
    image_model = init_from_ckpt(cavp_ckpt, image_model)
    return image_model, image_feat_dim


def init_talknet(talknet_ckpt=None):
    from visual_encoders.visual_encoder import visualFrontend

    talknet_ckpt = _require_path(talknet_ckpt or os.environ.get("TALKNET_CKPT"), "TALKNET_CKPT")
    image_feat_dim = 64
    image_model = visualFrontend(image_feat_dim)
    image_model.init_from_ckpt(talknet_ckpt)
    return image_model, image_feat_dim


def init_visual_encoder(visual_encoder_type, cavp_ckpt=None, talknet_ckpt=None):
    if visual_encoder_type == "cavp":
        image_model, image_feat_dim = init_cavp(cavp_ckpt=cavp_ckpt)
    elif visual_encoder_type == "talknet":
        image_model, image_feat_dim = init_talknet(talknet_ckpt=talknet_ckpt)
    elif visual_encoder_type == "both":
        image_model_cavp, image_feat_dim_cavp = init_cavp(cavp_ckpt=cavp_ckpt)
        image_model_talknet, image_feat_dim_talknet = init_talknet(talknet_ckpt=talknet_ckpt)
        image_model = torch.nn.ModuleList([image_model_cavp, image_model_talknet])
        image_feat_dim = [image_feat_dim_cavp, image_feat_dim_talknet]
    else:
        raise ValueError(f"Unsupported visual encoder type: {visual_encoder_type}")

    image_model.eval()
    for param in image_model.parameters():
        param.requires_grad = False

    return image_model, image_feat_dim


@torch.no_grad()
def forward_video(image_model, input_batch, visual_encoder_type):
    if visual_encoder_type == "both":
        image_feats_cavp = image_model[0].encode_video(input_batch[0] / 255.0, normalize=True, pool=False)
        image_feats_talknet = image_model[1].encode_video(input_batch[1] / 255.0, normalize=True, pool=False)
        return (image_feats_cavp, image_feats_talknet)

    input_batch = input_batch / 255.0
    if visual_encoder_type == "cavp":
        return image_model.encode_video(input_batch, normalize=True, pool=False)
    if visual_encoder_type == "talknet":
        return image_model.encode_video(input_batch)
    raise ValueError(f"Unsupported visual encoder type: {visual_encoder_type}")
