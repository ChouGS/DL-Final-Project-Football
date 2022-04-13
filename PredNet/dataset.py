from torch.utils.data import Dataset
import torch

class PredictorDataset(Dataset):
    def __init__(self, data) -> None:
        super(PredictorDataset, self).__init__()
        self.data = torch.Tensor(data)
    
    def __getitem__(self, index):
        return self.data[index, :67], self.data[index, 67:71], self.data[index, 71:]

    def __len__(self):
        return self.data.shape[0]
