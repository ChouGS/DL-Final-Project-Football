from torch.utils.data import Dataset
import torch
import numpy as np

class PredictorDataset(Dataset):
    def __init__(self, data) -> None:
        super(PredictorDataset, self).__init__()
        self.data = torch.Tensor(data)
    
    def __getitem__(self, index):
        # IMPORTANT: Data structure (11 player game as example)
        # 0-2: (x, y, dist) of teammate 1, ... till 30-32
        # 33-35: (x, y, dist) of rival 1, ..., till 63-65
        # 66-68: (x, y, dist) of ball
        # 69-70: (x, y) of self
        # 71-72: (vx, vy) of last tick
        # 73-76: self next decision (dx, dy, vx, vy)
        # 77: final x label
        # 78: score label
        # 79: touchdown or not
        return self.data[index, :71], self.data[index, 73:77], self.data[index, 78:79], self.data[index, 79:].long()

    def __len__(self):
        return self.data.shape[0]

class AttentionDataset(Dataset):
    def __init__(self, data) -> None:
        super(AttentionDataset, self).__init__()
        self.data = torch.Tensor(data)
    
    def __getitem__(self, index):
        # IMPORTANT: Data structure (11 player game as example)
        # 0-2: (x, y, dist) of teammate 1, ... till 30-32
        # 33-35: (x, y, dist) of rival 1, ..., till 63-65
        # 66-68: (x, y, dist) of ball
        # 69-70: (x, y) of self
        # 71-72: (vx, vy) of last tick
        # 73-76: self next decision (dx, dy, vx, vy)
        # 77: final x label
        # 78: score label
        # 79: touchdown or not
        att_feature = self.data[index, :69].reshape(23, 3)
        att_feature = torch.cat([att_feature, torch.zeros(23, 3)], 1)
        att_feature[:, 2:4] = self.data[index, 69:71]
        att_feature[:, 4:] = self.data[index, 75:77]
        return att_feature, self.data[index, 73:77], self.data[index, 78:79], self.data[index, 79:].long()

    def __len__(self):
        return self.data.shape[0]
