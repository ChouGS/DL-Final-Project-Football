from torch.utils.data import Dataset
import torch
import numpy as np

class PredictorDataset(Dataset):
    def __init__(self, data) -> None:
        super(PredictorDataset, self).__init__()
        self.data = torch.Tensor(data)
    
    def __getitem__(self, index):
        return self.data[index, :71], self.data[index, 71:75], self.data[index, 75:76], self.data[index, 77:].long()

    def __len__(self):
        return self.data.shape[0]

class AttentionDataset(Dataset):
    def __init__(self, data) -> None:
        super(AttentionDataset, self).__init__()
        self.data = torch.Tensor(data)
    
    def __getitem__(self, index):
        att_feature = self.data[index, :69].reshape(23, 3)
        att_feature = torch.cat([att_feature, torch.zeros(23, 3)], 1)
        att_feature[:, 2:4] = self.data[index, 69:71]
        att_feature[:, 4:] = self.data[index, 73:75]
        return att_feature, self.data[index, 71:75], self.data[index, 76:77], self.data[index, 77:].long()

    def __len__(self):
        return self.data.shape[0]