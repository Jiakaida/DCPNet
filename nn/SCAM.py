
import torch
import torch.nn as nn
from nn.common import ConvBNReLU


class DB(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(ConvBNReLU(channels, channels),
                                  ConvBNReLU(channels, channels))
        
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(channels, channels, k=1, p=0),
            ConvBNReLU(channels, channels, k=1, p=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        detail = self.conv(x)
        return x + self.gate(detail) * detail


class SCAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.projections = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 3, padding=1) for _ in range(3)
        ])
        self.context = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=d, dilation=d)
            for d in (1, 3, 5)
        ])
        self.reductions = nn.ModuleList([
            ConvBNReLU(out_channels * 2, out_channels) for _ in range(3)
        ])
        self.ba = DB(in_channels)
        self.ba_proj = ConvBNReLU(in_channels, out_channels)
        self.out = ConvBNReLU(out_channels, out_channels)

    def forward(self, x):
        branches = []
        for projection, context, reduction in zip(self.projections, self.context, self.reductions):
            projected = projection(x)
            branches.append(reduction(torch.cat([projected, context(projected)], dim=1)))
        return self.out(sum(branches) + self.ba_proj(self.ba(x)))


class MultiScaleSCAM(nn.Module):
    def __init__(self, in_channels, mid_ch=32):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError('SCAM requires four backbone feature scales')
        self.reduce = nn.ModuleList([SCAM(c, mid_ch) for c in in_channels])

    def forward(self, features):
        if len(features) != 4:
            raise ValueError('SCAM requires four backbone feature scales')
        return [block(feature) for block, feature in zip(self.reduce, features)]
