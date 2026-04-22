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
    
    def init_from_ckpt(self, path=None):
        if path is None:
            raise ValueError("TalkNet checkpoint path must be provided. Set TALKNET_CKPT env var.")
        model = torch.load(path, map_location="cpu")
        if "state_dict" in list(model.keys()):
            model = model["state_dict"]
        new_state_dict = model


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

