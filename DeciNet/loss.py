import torch
import torch.nn as nn

class DirectionLoss(nn.Module):
    def __init__(self, base=0.1) -> None:
        super(DirectionLoss, self).__init__()

    def forward(self, dir1, dir2):
        cos_sim = torch.cosine_similarity(dir1, dir2, dim=1).flatten()
        return torch.mean(1 / 1.1 + cos_sim)

class VeloLoss(nn.Module):
    def __init__(self, base=10, max_v=10) -> None:
        super().__init__()
        self.base = base
        self.max_v = max_v
    
    def forward(self, v):
        v_val = torch.norm(v)
        v_diff = v_val - self.max_v
        base = torch.ones_like(v_diff) * self.base
        return torch.mean(torch.pow(base, v_diff))
