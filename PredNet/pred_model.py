import torch
import torch.nn as nn
from collections import OrderedDict

class PredNet(nn.Module):
    def __init__(self, latent_dims, feature_dim=67) -> None:
        super(PredNet, self).__init__()
        latent_dims = [feature_dim] + latent_dims
        self.net = []
        for i in range(1, len(latent_dims)):
            # self.net.append((f'bn{i+1}', nn.BatchNorm1d(latent_dims[i-1])))
            self.net.append((f'fc{i+1}', nn.Linear(latent_dims[i-1], latent_dims[i])))
            self.net.append((f'relu{i+1}', nn.LeakyReLU(0.2)))
        self.net.append(('fc_outp', nn.Linear(latent_dims[-1], 1)))
        self.net = nn.Sequential(OrderedDict(self.net))

    def forward(self, x):
        return self.net(x)
