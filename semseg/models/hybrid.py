import torch
from torch import Tensor
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import SegFormerHead
from torch import nn

class ReverseAttention(nn.Module):
    def __init__(self, in_channel, out_channel, num_classes):
        super(ReverseAttention, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1)
        self.convs = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, 1, 3, padding=1),
        )
        self.num_classes = num_classes
        self.channel = out_channel

    def forward(self, x, y):
        batch_size = x.shape[0]
        x = self.convert(x)
        for i in range(self.num_classes):
            x_i = x[:, :, :, :]  # shape [batch_size, out_channel, height, width]
            y_i = y[:, i:i+1, :, :]  # shape [batch_size, 1, height, width]
            a_i = -1*(torch.sigmoid(y_i)) + 1
            x_i = a_i.expand(-1, self.channel, -1, -1).mul(x_i)
            y_i = y_i + self.convs(x_i)
            y[:, i:i+1, :, :] = y_i
        return y
    

class HyBrid(BaseModel):
    def __init__(self, backbone: str = 'MSCANet', num_classes: int = 2, channels = 256) -> None:
        super().__init__(backbone, num_classes)
        self.decode_head = SegFormerHead(self.backbone.channels, channels, num_classes)
        self.apply(self._init_weights)
        
        self.ra2 = ReverseAttention(64, channels, num_classes)
        self.ra3 = ReverseAttention(160, channels, num_classes)
        self.ra4 = ReverseAttention(256, channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x_size = x.size()[2:]
        y = self.backbone(x)

        # apply RA module for each feature blocks
        x1, x2, x3, x4 = y
        x2_size = x2.size()[2:]
        x3_size = x3.size()[2:]
        x4_size = x4.size()[2:]

        # decode head
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        # return y
        # compute 
        y5_4 = F.interpolate(y, size=x4_size, mode='bilinear', align_corners=True)
        y4 = self.ra4(x4, y5_4)
        score4 = F.interpolate(y4, x_size, mode='bilinear', align_corners=True)

        y4_3 = F.interpolate(y4, x3_size, mode='bilinear', align_corners=True)
        y3 = self.ra3(x3, y4_3)
        score3 = F.interpolate(y3, x_size, mode='bilinear', align_corners=True)

        y3_2 = F.interpolate(y3, x2_size, mode='bilinear', align_corners=True)	
        y2 = self.ra2(x2, y3_2)
        score2 = F.interpolate(y2, x_size, mode='bilinear', align_corners=True)

        return y, score4, score3, score2


if __name__ == '__main__':
    model = HyBrid('MSCANet')
    # model.load_state_dict(torch.load('checkpoints/pretrained/segformer/segformer.b0.ade.pth', map_location='cpu'))
    x = torch.zeros(1, 3, 512, 512)
    y = model(x)
    print(y.shape)