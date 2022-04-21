from torch.utils.data import Dataset
import torch

class PredictorDataset(Dataset):
    def __init__(self, data) -> None:
        super(PredictorDataset, self).__init__()
        self.data = torch.Tensor(data)
    
    def __getitem__(self, index):
        return self.data[index, :71], self.data[index, 71:75], self.data[index, 75:76], self.data[index, 76:].long()

    def __len__(self):
        return self.data.shape[0]

class AttentionDataset(Dataset):
    def __init__(self, data) -> None:
        super(AttentionDataset, self).__init__()
        self.data = torch.Tensor(data)
    
    def __getitem__(self, index):
        return self.data[index, :69].reshape(23, 3), self.data[index, 71:75], self.data[index, 75:76], self.data[index, 76:].long()

    def __len__(self):
        return self.data.shape[0]