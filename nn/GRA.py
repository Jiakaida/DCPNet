
import torch
import torch.nn as nn
import torch.nn.functional as F
from nn.common import ConvBNReLU

class GRA(nn.Module):

    def __init__(self, mid_ch=64, n_splits=4):
        super().__init__()
        assert mid_ch % n_splits == 0, "mid_ch 必须能被 n_splits 整除"
        self.n_splits = n_splits
        self.group_ch = mid_ch // n_splits          
        
        fused_ch = n_splits * (self.group_ch + 1)
        self.fuse = ConvBNReLU(fused_ch, mid_ch, k=3, p=1)

    def forward(self, rgb, rev_att):
        
        groups = torch.chunk(rgb, self.n_splits, dim=1)   

        
        interleaved = []
        for g in groups:
            interleaved.append(g)
            interleaved.append(rev_att)                   

        
        x = torch.cat(interleaved, dim=1)                 
        return self.fuse(x)
