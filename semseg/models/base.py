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
            # self.backbone = eval(backbone)(in_channnels=3, embed_dims=[64, 128, 320, 512],
            #      ffn_ratios=[8, 8, 4, 4], depths=[3, 5, 27, 3], num_stages=4,
            #      ls_init_val=1e-2, drop_path=0.0)
            # self.backbone = eval(backbone)(in_channnels=3, embed_dims=[32, 64, 160, 256],
            #      ffn_ratios=[8, 8, 4, 4], depths=[3, 3, 5, 2], num_stages=4,
            #      ls_init_val=1e-2, drop_path=0.0)
            # self.backbone = eval(backbone)(in_channnels=3, embed_dims=[64, 128, 320, 512],
            #      ffn_ratios=[8, 8, 4, 4], depths=[3, 3, 12, 3], num_stages=4,
            #      ls_init_val=1e-2, drop_path=0.0)
            # self.backbone = eval(backbone)(in_channnels=3, embed_dims=[64, 128, 320, 320],
            #      ffn_ratios=[8, 8, 4, 4], depths=[3, 5, 27, 3], num_stages=4,
            #      ls_init_val=1e-2, drop_path=0.0)
            self.backbone = eval(backbone)('S36')
            self.backbone.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            fan_in // m.groups
            std = math.sqrt(2.0 / fan_in)
            m.weight.data.normal_(0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str = None) -> None:
        print(pretrained)
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)