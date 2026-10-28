
import torch
import torch.nn as nn
from model.pvtv2 import pvt_v2_b2
from model.DFPD import DFPD


class DCPNet(nn.Module):
    def __init__(self, channel=32, kernel_size=3, reduction=4, bias=False,
                 act=None, n_resblocks=2, iteration=3,
                 pretrained_path='/root/autodl-tmp/CCD_Net/pvt_v2_b2.pth'):
        super().__init__()
        self.rgb_backbone = pvt_v2_b2()
        if pretrained_path is not None:
            saved = torch.load(pretrained_path, map_location='cpu')
            current = self.rgb_backbone.state_dict()
            current.update({k: v for k, v in saved.items() if k in current})
            self.rgb_backbone.load_state_dict(current)
        self.head = DFPD(mid_ch=channel)

    def forward(self, rgb):
        return self.head(self.rgb_backbone(rgb), output_size=rgb.shape[-2:])
