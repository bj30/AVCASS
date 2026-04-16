import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetLayer(nn.Module):

    """
    A ResNet layer used to build the ResNet network.
    Architecture:
    --> conv-bn-relu -> conv -> + -> bn-relu -> conv-bn-relu -> conv -> + -> bn-relu -->
     |                        |   |                                    |
     -----> downsample ------>    ------------------------------------->
    """

    def __init__(self, inplanes, outplanes, stride):
        super(ResNetLayer, self).__init__()
        self.conv1a = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1a = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2a = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride
        # if self.stride != 1:
        self.downsample = nn.Conv2d(inplanes, outplanes, kernel_size=(1,1), stride=stride, bias=False)
        self.outbna = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)

        self.conv1b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1b = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        self.conv2b = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.outbnb = nn.BatchNorm2d(outplanes, momentum=0.01, eps=0.001)
        return


    def forward(self, inputBatch):
        batch = F.relu(self.bn1a(self.conv1a(inputBatch)))
        batch = self.conv2a(batch)
        # if self.stride == 1:
        #     residualBatch = inputBatch
        # else:
        residualBatch = self.downsample(inputBatch)
        batch = batch + residualBatch
        intermediateBatch = batch
        batch = F.relu(self.outbna(batch))

        batch = F.relu(self.bn1b(self.conv1b(batch)))
        batch = self.conv2b(batch)
        residualBatch = intermediateBatch
        batch = batch + residualBatch
        outputBatch = F.relu(self.outbnb(batch))
        return outputBatch



class ResNet(nn.Module):

    """
    An 18-layer ResNet architecture.
    """

    def __init__(self, feat_dim=512):
        super(ResNet, self).__init__()
        # if feat_dim==512:
        #     self.layer1 = ResNetLayer(64, 64, stride=1)
        #     self.layer2 = ResNetLayer(64, 128, stride=2)
        #     self.layer3 = ResNetLayer(128, 256, stride=2)
        #     self.layer4 = ResNetLayer(256, 512, stride=2)
        # elif feat_dim==128:
        self.layer1 = ResNetLayer(64, 64, stride=1)
        self.layer2 = ResNetLayer(64, 64, stride=2)
        self.layer3 = ResNetLayer(64, 128, stride=2)
        self.layer4 = ResNetLayer(128, 128, stride=2)
        # elif feat_dim==64:
        #     self.layer1 = ResNetLayer(64, 64, stride=1)
        #     self.layer2 = ResNetLayer(64, 64, stride=1)
        #     self.layer3 = ResNetLayer(64, 64, stride=1)
        #     self.layer4 = ResNetLayer(64, 64, stride=1)
        self.avgpool = nn.AvgPool2d(kernel_size=(4,4), stride=(1,1))
        
        return


    def forward(self, inputBatch):
        batch = self.layer1(inputBatch)
        batch = self.layer2(batch)
        batch = self.layer3(batch)
        batch = self.layer4(batch)
        outputBatch = self.avgpool(batch)
        return outputBatch


class GlobalLayerNorm(nn.Module):
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + 1e-8, 0.5) + self.beta
        return gLN_y



class DSConv1d(nn.Module):
    def __init__(self, feat_dim=64):
        super(DSConv1d, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(feat_dim*2),
            nn.Conv1d(feat_dim*2, feat_dim*2, 3, stride=1, padding=1,dilation=1, groups=feat_dim*2, bias=False),
            nn.PReLU(),
            GlobalLayerNorm(feat_dim*2),
            nn.Conv1d(feat_dim*2, feat_dim*2, 1, bias=False),
            )

    def forward(self, x):
        out = self.net(x)
        return out + x

class visualTCN(nn.Module):
    def __init__(self, feat_dim):
        super(visualTCN, self).__init__()
        stacks = []        
        for x in range(5):
            stacks += [DSConv1d(feat_dim)]
        self.net = nn.Sequential(*stacks) # Visual Temporal Network V-TCN

    def forward(self, x):
        out = self.net(x)
        return out

class visualConv1D(nn.Module):
    def __init__(self, feat_dim):
        super(visualConv1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feat_dim*2, feat_dim, 5, stride=1, padding=2),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            # nn.Conv1d(feat_dim, feat_dim, 1),
            )

    def forward(self, x):
        out = self.net(x)
        return out

