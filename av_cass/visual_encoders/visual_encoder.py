##
# ResNet18 Pretrained network to extract lip embedding
# This code is modified based on https://github.com/TaoRuijie/TalkNet-ASD/blob/main/model/visualEncoder.py
##

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from visual_encoders.talknet import ResNet, visualTCN, visualConv1D
import pytorch_lightning as pl

class visualFrontend(pl.LightningModule):
    """
    A visual feature extraction module. Generates a {feat_dim}-dim feature vector per video frame.
    Architecture: A 3D convolution block followed by an 18-layer ResNet.
    Adopted from TalkNet (https://github.com/TaoRuijie/TalkNet-ASD/tree/main)
    """

    def __init__(self, feat_dim=64):
        super().__init__()
        self.frontend3D = nn.Sequential(
                            nn.Conv3d(1, feat_dim, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False),
                            nn.BatchNorm3d(feat_dim, momentum=0.01, eps=0.001),
                            nn.ReLU(),
                            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
                        )
        self.resnet = ResNet(feat_dim)
        
        self.visualTCN       = visualTCN(feat_dim)
        self.visualConv1D    = visualConv1D(feat_dim)
        self.feat_dim = feat_dim
    
    def init_from_ckpt(self, path='/mnt/bear2/users/syun/avdiffuss_epoch30_visual_encoder.ckpt'):
        model = torch.load(path, map_location="cpu")
        if "state_dict" in list(model.keys()):
            model = model["state_dict"]
        new_state_dict = model
        # Remove: module prefix
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     if 'denoiser_net.visual_encoder.' in k:
        #         new_k = k.replace('denoiser_net.visual_encoder.', '')
        #         new_state_dict[new_k] = v

        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def forward(self, inputBatch):
        x = self.encode_video(inputBatch)
        return x

    def encode_video(self, inputBatch, **kwargs):
        # inputBatch.shape = (B, T=204, W=112, H=112)
        if inputBatch.ndim == 4: # expected to be default
            B, T, W, H = inputBatch.shape
            inputBatch = inputBatch.view(B*T, 1, 1, W, H)
        elif inputBatch.ndim==5:
            B, T, C, H, W = inputBatch.shape
            assert C == 1, "please load grayscale images for 'talknet' encoder / "+str((B, T, C, H, W))
            inputBatch = inputBatch.view(B*T, 1, 1, W, H)


        # inputBatch = (inputBatch / 255 - 0.4161) / 0.1688 
        inputBatch = (inputBatch - 0.4161) / 0.1688 # because it's already divided by 255 in Model.forward_video()

        inputBatch = inputBatch.transpose(0, 1).transpose(1, 2)

        batchsize = inputBatch.shape[0]
        batch = self.frontend3D(inputBatch)

        batch = batch.transpose(1, 2)
        batch = batch.reshape(batch.shape[0]*batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4])
        outputBatch = self.resnet(batch)
        
        x = outputBatch.view(B, T, self.feat_dim*2)
        
        x = x.transpose(1,2)
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        x = x.transpose(1,2)
        return x


'''
# from main.visual.shared import norm
class visualFrontend(nn.Module):
    """
    A visual feature extraction module. Generates a {feat_dim}-dimensional feature vector per video frame.
    Architecture: A 3D convolution block followed by an 18-layer ResNet.
    """

    def __init__(self, feat_dim=512, visual_encoder_type='cavp'):
        super(visualFrontend, self).__init__()
        self.feat_dim = feat_dim
        self.visual_encoder_type = visual_encoder_type
        if visual_encoder_type=='talknet':
            # default feat dim = 64
            self.frontend3D = nn.Sequential(
                                nn.Conv3d(1, 64, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3), bias=False),
                                nn.BatchNorm3d(64, momentum=0.01, eps=0.001),
                                nn.ReLU(),
                                nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
                            )
            self.resnet = ResNet(feat_dim)
            self.visualTCN    = visualTCN(feat_dim)
            self.visualConv1D = visualConv1D(feat_dim)
            
            

            
        elif 'dinov2' in visual_encoder_type:
            load_size = 224  # * 3
            if visual_encoder_type=='dinov2_noT':
                hidden_dim = 1024
            elif visual_encoder_type=='dinov2':
                hidden_dim = 512
            self.transform = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            self.encoder = DINOv2Featurizer()
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            self.projector = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # [B,T,C=768,1,1]
                nn.Flatten(start_dim=-3),  # [B,T,768]
                nn.Linear(768, hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, feat_dim) # [B,T,feat_dim]
            )
        elif visual_encoder_type=='cavp':
            self.fps = 4
            # cavp_config_path = "./config/Stage1_CAVP.yaml"              #  CAVP Config
            config = OmegaConf.create({"model": {"target":"model.cavp_model.CAVP_Inference", "params":{"video_encode": "Slowonly_pool",
                                    "spec_encode": "cnn14_pool", "embed_dim": 512, "video_pretrained": True, "audio_pretrained": True}}})
            self.encoder = instantiate_from_config(config.model) #.to(device)
            # Loading Model from:
            ckpt_path = "/mnt/bear3/users/syun/huggingface_models/diff_foley_ckpt/cavp_epoch66.ckpt"      #  CAVP Ckpt
            print("Loading CAVP Model from: {}".format(ckpt_path))
            self.init_first_from_ckpt(ckpt_path)
            """
            load_size = 224
            # Transform:
            self.img_transform = transforms.Compose([
                transforms.Resize(load_size),
                transforms.ToTensor(),
            ])"""
        else:
            raise ValueError(f"Visual encoder type is not defined for {visual_encoder_type}")
            
    def init_first_from_ckpt(self, path):
        model = torch.load(path, map_location="cpu")
        if "state_dict" in list(model.keys()):
            model = model["state_dict"]
        # Remove: module prefix
        new_model = {}
        for key in model.keys():
            new_key = key.replace("module.","")
            new_model[new_key] = model[key]
        missing, unexpected = self.encoder.load_state_dict(new_model, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")


    def forward(self, inputBatch):
        if self.visual_encoder_type=='talknet':
            # inputBatch.shape = (B, T=204, W=112, H=112)
            if inputBatch.ndim!=4:
                inputBatch.unsqueeze(0)
            B, T, W, H = inputBatch.shape
            inputBatch = inputBatch.view(B*T, 1, 1, W, H)
            inputBatch = (inputBatch / 255 - 0.4161) / 0.1688

            inputBatch = inputBatch.transpose(0, 1).transpose(1, 2)
            batch = self.frontend3D(inputBatch)

            batch = batch.transpose(1, 2)
            batch = batch.reshape(batch.shape[0]*batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4])
            outputBatch = self.resnet(batch)
            outputBatch = outputBatch.reshape(B, -1, self.feat_dim) # (B, T, F)
            return outputBatch

        elif self.visual_encoder_type=='cavp':
            # inputBatch.shape = (B, T, C=3, W=224, H=224)
            if inputBatch.ndim==4:
                inputBatch.unsqueeze(0)
            B,T,C,W,H = inputBatch.shape
            outputBatch = self.encoder.encode_video(inputBatch, normalize=True, pool=False) # (B, T, C=512)
            return outputBatch
            
        else:
            # inputBatch.shape = (B, T, C=3, W=224, H=224)
            if inputBatch.ndim==4:
                inputBatch.unsqueeze(0)
            B,T,C,W,H = inputBatch.shape
            inputBatch = inputBatch.view(B*T, C, W, H)
            transformed_input = self.transform(inputBatch)
            with torch.no_grad():
                outputBatch = self.encoder(transformed_input, include_cls=False)
            outputBatch = outputBatch.reshape(B, T, -1, W//14, H//14)
            outputBatch = self.projector(outputBatch) # [B, T, feat_dim]
            return outputBatch
'''