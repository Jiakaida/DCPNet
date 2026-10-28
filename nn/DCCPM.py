
import torch
import torch.nn as nn
import torch.nn.functional as F
from nn.common import ConvBNReLU


class DynamicMultiKernelBranch(nn.Module):
    
    def __init__(self, channels, out_channels, kernel_sizes=(5, 7, 9, 11)):
        super().__init__()
        self.dw_conv = nn.ModuleList([
            nn.Conv2d(channels, channels, k, padding=k // 2, groups=channels)
            for k in kernel_sizes
        ])
        concat_channels = channels * (len(kernel_sizes) + 1)
        hidden = max(concat_channels // 16, 8)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(concat_channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, concat_channels, 1),
        )
        self.output = ConvBNReLU(concat_channels, out_channels, k=1, p=0)

    def forward(self, x):
        spatial = torch.cat([layer(x) for layer in self.dw_conv] + [x], dim=1)
        weight = torch.sigmoid(self.channel_mlp(F.adaptive_avg_pool2d(spatial, 1)))
        return self.output(spatial * weight)


class HaarWaveletDecomposition(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        filters = torch.tensor([
            [[1., 1.], [1., 1.]],
            [[1., -1.], [1., -1.]],
            [[1., 1.], [-1., -1.]],
            [[1., -1.], [-1., 1.]],
        ]).unsqueeze(1) / 2.0
        self.register_buffer('weights', filters.repeat(channels, 1, 1, 1), persistent=False)

    def forward(self, x):
        pad_h, pad_w = x.shape[-2] % 2, x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        subbands = F.conv2d(x, self.weights, stride=2, groups=self.channels)
        b, _, h, w = subbands.shape
        subbands = subbands.reshape(b, self.channels, 4, h, w)
        return tuple(subbands[:, :, i] for i in range(4))

    def inverse(self, subbands, output_size=None):
        if len(subbands) != 4:
            raise ValueError('IDWT requires LL, LH, HL and HH')
        packed = torch.stack(subbands, dim=2)
        b, c, _, h, w = packed.shape
        if c != self.channels:
            raise ValueError('IDWT channel count does not match Haar filters')
        reconstructed = F.conv_transpose2d(
            packed.reshape(b, c * 4, h, w), self.weights,
            stride=2, groups=self.channels,
        )
        if output_size is not None:
            reconstructed = reconstructed[..., :output_size[0], :output_size[1]]
        return reconstructed


class WaveletFrequencyBranch(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.analysis = HaarWaveletDecomposition(in_channels)
        self.subband_conv = ConvBNReLU(in_channels * 4, out_channels * 4)
        self.synthesis = HaarWaveletDecomposition(out_channels)

    def forward(self, x):
        subbands = self.analysis(x)
        processed = self.subband_conv(torch.cat(subbands, dim=1))
        return self.synthesis.inverse(processed.chunk(4, dim=1), x.shape[-2:])


class DualDomainFeatures(nn.Module):
    
    def __init__(self, in_channels, mid_ch=32, kernel_sizes=(5, 7, 9, 11)):
        super().__init__()
        if isinstance(in_channels, int):
            in_channels = (in_channels,) * 4
        if len(in_channels) != 4:
            raise ValueError('DCCPM requires four backbone feature scales')
        self.input_proj = nn.ModuleList([
            nn.Conv2d(c, mid_ch, 1) for c in in_channels
        ])
        self.spatial_focus = DynamicMultiKernelBranch(mid_ch * 4, mid_ch, kernel_sizes)
        self.frequency = WaveletFrequencyBranch(mid_ch * 4, mid_ch)
        hidden = max(mid_ch // 4, 8)
        
        self.branch_gate = nn.Sequential(
            nn.Conv2d(mid_ch * 2, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, mid_ch * 2, 1),
        )

    def forward(self, features):
        if len(features) != 4:
            raise ValueError('DCCPM requires four backbone feature scales')
        target = features[0].shape[-2:]
        aligned = [proj(F.interpolate(feat, size=target, mode='bilinear', align_corners=False))
                   for proj, feat in zip(self.input_proj, features)]
        fused = torch.cat(aligned, dim=1)
        spatial = self.spatial_focus(fused)
        frequency = self.frequency(fused)
        combined = torch.cat([spatial, frequency], dim=1)
        weights = (
            torch.sigmoid(self.branch_gate(F.adaptive_avg_pool2d(combined, 1)))
            + torch.sigmoid(self.branch_gate(F.adaptive_max_pool2d(combined, 1)))
        )
        spatial_weight, frequency_weight = weights.chunk(2, dim=1)
        return spatial * spatial_weight + frequency * frequency_weight


class DCCPM(nn.Module):
    def __init__(self, in_channels=(64, 128, 320, 512), mid_ch=32,
                 kernel_sizes=(5, 7, 9, 11)):
        super().__init__()
        self.dynamic_focuse = DualDomainFeatures(in_channels, mid_ch, kernel_sizes)
        self.head = nn.Conv2d(mid_ch, 1, 1)

    def forward(self, features):
        feature = self.dynamic_focuse(features)
        return self.head(feature), feature
