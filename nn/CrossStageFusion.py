
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channel, out_channel, 5, padding=2)
        self.conv3 = nn.Conv2d(in_channel, out_channel, 7, padding=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        
        return self.relu(self.conv1(x)) + self.relu(self.conv2(x)) + self.relu(self.conv3(x))


class CrossStageChannelAttention(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.channels = in_channel
        hidden = max(in_channel // 4, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channel * 4, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channel * 2, 1),
        )

    def forward(self, t1, t2):
        pooled = torch.cat([
            F.adaptive_avg_pool2d(t1, 1), F.adaptive_max_pool2d(t1, 1),
            F.adaptive_avg_pool2d(t2, 1), F.adaptive_max_pool2d(t2, 1),
        ], dim=1)
        first, second = torch.sigmoid(self.mlp(pooled)).chunk(2, dim=1)
        return torch.stack([first, second], dim=0)


class CrossStageSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 2, 7, padding=3)

    def forward(self, t1, t2):
        pooled = torch.cat([
            t1.mean(dim=1, keepdim=True), t1.amax(dim=1, keepdim=True),
            t2.mean(dim=1, keepdim=True), t2.amax(dim=1, keepdim=True),
        ], dim=1)
        first, second = torch.sigmoid(self.conv(pooled)).chunk(2, dim=1)
        return torch.stack([first, second], dim=0)


class ChannelSpatialFusion(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.channel_attention = CrossStageChannelAttention(in_channel)
        self.spatial_attention = CrossStageSpatialAttention()

    def forward(self, t1, t2):
        weights = self.channel_attention(t1, t2) + self.spatial_attention(t1, t2)
        return weights[0] * t1 + weights[1] * t2


class CrossStageFusion(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.multi_scale_extractor = MultiScaleFeatureExtractor(in_channel, in_channel)
        self.csf = ChannelSpatialFusion(in_channel)

    def forward(self, current_feature, previous_feature):
        current = self.multi_scale_extractor(current_feature)
        previous = self.multi_scale_extractor(previous_feature)
        return self.csf(current, previous)
