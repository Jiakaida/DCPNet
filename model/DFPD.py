
import torch.nn as nn
import torch.nn.functional as F
from nn.SCAM import MultiScaleSCAM
from nn.DCCPM import DCCPM
from nn.GRA import GRA
from nn.CrossStageFusion import CrossStageFusion


class DFPDStage(nn.Module):
    
    def __init__(self, mid_ch=32, n_splits=4):
        super().__init__()
        self.rgb_progressive = GRA(mid_ch, n_splits)
        self.bfm = CrossStageFusion(mid_ch)
        self.head = nn.Conv2d(mid_ch, 1, 1)

    def forward(self, rgb, prev_map, prev_feat):
        size = rgb.shape[-2:]
        prev_map_up = F.interpolate(prev_map, size=size, mode='bilinear', align_corners=False)
        rev_att = 1.0 - prev_map_up.sigmoid()
        current = self.rgb_progressive(rgb, rev_att)
        previous = F.interpolate(prev_feat, size=size, mode='bilinear', align_corners=False)
        fused = self.bfm(current, previous)
        return self.head(fused), fused


class DFPD(nn.Module):
    def __init__(self, rgb_in_channels=(64, 128, 320, 512), mid_ch=32,
                 n_splits=4, out_size=None):
        super().__init__()
        self.out_size = out_size
        self.rgb_unify = MultiScaleSCAM(rgb_in_channels, mid_ch)
        self.coarse = DCCPM(rgb_in_channels, mid_ch)
        
        self.ra_stages = nn.ModuleList([DFPDStage(mid_ch, n_splits) for _ in range(4)])

    def _resize(self, x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    def forward(self, rgb_feats, output_size=None):
        if len(rgb_feats) != 4:
            raise ValueError('DFPD requires four backbone feature scales')
        coarse_map, coarse_feat = self.coarse(rgb_feats)
        context = self.rgb_unify(rgb_feats)
        prev_map, prev_feat = coarse_map, coarse_feat
        stage_maps = [None] * 4
        
        for idx in (3, 2, 1, 0):
            prev_map, prev_feat = self.ra_stages[idx](context[idx], prev_map, prev_feat)
            stage_maps[idx] = prev_map
        target = output_size if output_size is not None else self.out_size
        if target is None:
            target = context[0].shape[-2:]
        if isinstance(target, int):
            target = (target, target)
        side = [self._resize(prediction, target) for prediction in stage_maps]
        return {'pred': side[0], 'coarse': self._resize(coarse_map, target), 'side': side}
