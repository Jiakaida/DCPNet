

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1, relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ConvBNPReLU(nn.Module):

    def __init__(self,

                 in_channels,

                 out_channels,

                 kernel_size=3,

                 stride=1,

                 padding=1,

                 dilation=1):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(

                in_channels,

                out_channels,

                kernel_size,

                stride,

                padding,

                dilation=dilation,

                bias=False,

            ),

            nn.BatchNorm2d(out_channels),

            nn.PReLU(out_channels),

        )

    def forward(self, x):

        return self.block(x)
