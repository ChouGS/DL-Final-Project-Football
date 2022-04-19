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

class PredAE(nn.Module):
    def __init__(self, encoder_structure, latent_dim=16, feature_dim=67):
        super(PredAE, self).__init__()
        encoder_structure = [feature_dim] + encoder_structure + [latent_dim]
        decoder_structure = list(reversed(encoder_structure))

        self.encoder = []
        for i in range(1, len(encoder_structure) - 1):
            # self.net.append((f'bn{i+1}', nn.BatchNorm1d(latent_dims[i-1])))
            self.encoder.append((f'fc{i+1}', nn.Linear(encoder_structure[i-1], encoder_structure[i])))
            self.encoder.append((f'relu{i+1}', nn.LeakyReLU(0.2)))
        self.encoder.append(('fc_outp', nn.Linear(encoder_structure[-2], encoder_structure[-1])))
        self.encoder = nn.Sequential(OrderedDict(self.encoder))

        self.decoder = []
        for i in range(1, len(decoder_structure) - 1):
            # self.net.append((f'bn{i+1}', nn.BatchNorm1d(latent_dims[i-1])))
            self.decoder.append((f'fc{i+1}', nn.Linear(decoder_structure[i-1], decoder_structure[i])))
            self.decoder.append((f'relu{i+1}', nn.LeakyReLU(0.2)))
        self.decoder.append(('fc_outp', nn.Linear(decoder_structure[-2], decoder_structure[-1])))
        self.decoder = nn.Sequential(OrderedDict(self.decoder))

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec
