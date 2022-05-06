import torch
import torch.nn as nn

class DirectionLoss(nn.Module):
    def __init__(self) -> None:
        super(DirectionLoss, self).__init__()

    def forward(self, dir1, dir2):
        cos_sim = torch.cosine_similarity(dir1, dir2, dim=1).flatten()
        return torch.mean(1 / 1.1 + cos_sim)

class OOBLoss(nn.Module):
    def __init__(self, lbound=0, ubound=400) -> None:
        super(OOBLoss, self).__init__()
        self.lb = lbound
        self.ub = ubound
        self.margin = 20

    def forward(self, pos, v):
        ppos = pos + v * 0.05
        ppos = ppos[:, 1]
        return torch.mean(1 / (ppos + self.lb + self.margin) + 1 / (self.ub + self.margin - ppos))

class VeloLoss(nn.Module):
    def __init__(self, best_v=90) -> None:
        super().__init__()
        self.best_v = best_v

    def forward(self, v):
        v_val = torch.norm(v, dim=1)
        v_diff = v_val - self.best_v
        return torch.mean(v_diff * v_diff)
