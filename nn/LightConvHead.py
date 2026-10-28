import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DSConvBNReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class FeatureFusionModule(nn.Module):

    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        
        self.lateral_convs = nn.ModuleList([
            ConvBNReLU(in_ch, out_channels, kernel_size=1, padding=0)
            for in_ch in in_channels_list
        ])
        
        self.fusion_convs = nn.ModuleList([
            DSConvBNReLU(out_channels, out_channels)
            for _ in in_channels_list
        ])
        self.attention = CBAM(out_channels)

    def forward(self, features):
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        for i in range(len(laterals) - 2, -1, -1):
            h, w = laterals[i].shape[2:]
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=(h, w), mode='bilinear', align_corners=False
            )

        outputs = [conv(lat) for conv, lat in zip(self.fusion_convs, laterals)]

        target_size = outputs[0].shape[2:]
        fused = sum([
            F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
            if out.shape[2:] != target_size else out
            for out in outputs
        ])
        return self.attention(fused)


class BoundaryRefinementModule(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.edge_conv = nn.Sequential(
            DSConvBNReLU(channels, channels),
            DSConvBNReLU(channels, channels),
            nn.Conv2d(channels, 1, 1)
        )
        self.body_conv = nn.Sequential(
            DSConvBNReLU(channels, channels),
            DSConvBNReLU(channels, channels),
            nn.Conv2d(channels, 1, 1)
        )
        self.fuse_conv = nn.Sequential(
            DSConvBNReLU(channels + 2, channels),
            nn.Conv2d(channels, 1, 1)
        )

    def forward(self, x):
        edge = self.edge_conv(x)
        body = self.body_conv(x)

        fused = self.fuse_conv(torch.cat([x, edge, body], dim=1))

        return fused


class ReverseAttentionModule(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv1 = DSConvBNReLU(channels, channels)
        self.conv2 = DSConvBNReLU(channels, channels)
        self.conv3 = nn.Conv2d(channels, 1, 1)

    def forward(self, x, pred):
        reverse_attn = 1 - pred
        x = x * reverse_attn
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


class CODDetectionHeadLightweight(nn.Module):

    def __init__(
            self,
            in_channels_list=[768, 768, 768, 768],
            hidden_channels=256,
            num_stages=3
    ):
        super().__init__()
        self.num_stages = num_stages

        self.feature_fusion = FeatureFusionModule(in_channels_list, hidden_channels)

        self.initial_pred = nn.Sequential(
            DSConvBNReLU(hidden_channels, hidden_channels),
            DSConvBNReLU(hidden_channels, hidden_channels // 2),
            nn.Conv2d(hidden_channels // 2, 1, 1)
        )

        self.refine_stages = nn.ModuleList([
            ReverseAttentionModule(hidden_channels)
            for _ in range(num_stages)
        ])

        self.boundary_refine = BoundaryRefinementModule(hidden_channels)

        self.final_fuse = nn.Sequential(
            DSConvBNReLU(num_stages + 2, hidden_channels // 4),
            nn.Conv2d(hidden_channels // 4, 1, 1)
        )

    def forward(self, features, target_size=None):

        fused = self.feature_fusion(features)

        pred = self.initial_pred(fused)
        preds = [pred]

        for refine in self.refine_stages:
            pred_sigmoid = torch.sigmoid(pred)
            
            if pred_sigmoid.shape[2:] != fused.shape[2:]:
                pred_sigmoid = F.interpolate(
                    pred_sigmoid, size=fused.shape[2:],
                    mode='bilinear', align_corners=False
                )
            pred = refine(fused, pred_sigmoid)
            preds.append(pred)

        boundary_pred = self.boundary_refine(fused)

        all_preds = torch.cat(preds + [boundary_pred], dim=1)
        final_pred = self.final_fuse(all_preds)

        if target_size is not None:
            final_pred = F.interpolate(final_pred, size=target_size, mode='bilinear', align_corners=False)
            preds = [F.interpolate(p, size=target_size, mode='bilinear', align_corners=False) for p in preds]

        return {
            'pred': final_pred,
            'pred_stages': preds,
        }
