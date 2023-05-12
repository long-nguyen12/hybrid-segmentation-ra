import torch
import math
from torch import nn
from semseg.models.backbones import *
from semseg.models.layers import trunc_normal_


class BaseModel(nn.Module):
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 19) -> None:
        super().__init__()
        if "-" in backbone:
            backbone, variant = backbone.split('-')
            self.backbone = eval(backbone)(variant)
        else:
            self.backbone = eval(backbone)(in_channnels=3, embed_dims=[32, 64, 160, 256],
                 ffn_ratios=[8, 8, 4, 4], depths=[3,3,5,2], num_stages=4,
                 ls_init_val=1e-2, drop_path=0.0)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)