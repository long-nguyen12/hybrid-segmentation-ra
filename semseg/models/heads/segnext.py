from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn

from .hamburger import HamBurger
from ..bricks import SeprableConv2d, ConvRelu, ConvBNRelu, resize


class HamDecoder(nn.Module):
    '''SegNext'''
    def __init__(self, outChannels, config, enc_embed_dims=[64,128,320,512]):
        super().__init__()

        ham_channels = config['ham_channels']
        self.squeeze = ConvRelu(sum(enc_embed_dims), ham_channels)
        self.ham_attn = HamBurger(ham_channels, config)
        self.align = ConvRelu(ham_channels, 1)
       
    def forward(self, features):
        # features = features[1:] # drop stage 1 features b/c low level
        features = [resize(feature, size=features[0].shape[2:], mode='bilinear') for feature in features]
        x = torch.cat(features, dim=1)

        x = self.squeeze(x)
        x = self.ham_attn(x)
        x = self.align(x)       

        return x