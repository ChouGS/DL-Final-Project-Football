import torch
import torch.nn as nn
from collections import OrderedDict


class PredTD(nn.Module):
    '''
    Bi-classification MLP to predict touchdown label
    '''
    def __init__(self, cfg) -> None:
        super(PredTD, self).__init__()
        latent_dims = cfg.STRUCTURE
        feature_dim = cfg.IN_DIM
        latent_dims = [feature_dim] + latent_dims
        self.net = []
        for i in range(1, len(latent_dims)):
            if cfg.USE_BN:
                self.net.append((f'bn{i+1}', nn.BatchNorm1d(latent_dims[i-1])))
            self.net.append((f'fc{i+1}', nn.Linear(latent_dims[i-1], latent_dims[i])))
            self.net.append((f'relu{i+1}', nn.LeakyReLU(0.2)))
        self.net.append(('fc_outp', nn.Linear(latent_dims[-1], 2)))
        self.net.append(('softmax', nn.Softmax(1)))
        self.net = nn.Sequential(OrderedDict(self.net))

    def forward(self, x):
        return self.net(x)

class PredX(nn.Module):
    '''
    MLP network for X label prediction
    '''
    def __init__(self, cfg) -> None:
        super(PredX, self).__init__()
        latent_dims = cfg.STRUCTURE
        feature_dim = cfg.IN_DIM
        latent_dims = [feature_dim] + latent_dims
        self.net = []
        for i in range(1, len(latent_dims)):
            if cfg.USE_BN:
                self.net.append((f'bn{i+1}', nn.BatchNorm1d(latent_dims[i-1])))
            self.net.append((f'fc{i+1}', nn.Linear(latent_dims[i-1], latent_dims[i])))
            self.net.append((f'relu{i+1}', nn.LeakyReLU(0.2)))
        self.net.append(('fc_outp', nn.Linear(latent_dims[-1], 1)))
        self.net = nn.Sequential(OrderedDict(self.net))

    def forward(self, x):
        return self.net(x)

class PredAE(nn.Module):
    '''
    Autoencoder model to reduce feature dim
    '''
    def __init__(self, cfg):
        super(PredAE, self).__init__()
        latent_dim = cfg.LAT_D
        feature_dim = cfg.IN_DIM
        encoder_structure = cfg.STRUCTURE
        encoder_structure = [feature_dim] + encoder_structure + [latent_dim]
        decoder_structure = list(reversed(encoder_structure))

        # Encoder block
        self.encoder = []
        for i in range(1, len(encoder_structure) - 1):
            self.encoder.append((f'fc{i+1}', nn.Linear(encoder_structure[i-1], encoder_structure[i])))
            self.encoder.append((f'relu{i+1}', nn.LeakyReLU(0.2)))
        self.encoder.append(('fc_outp', nn.Linear(encoder_structure[-2], encoder_structure[-1])))
        self.encoder = nn.Sequential(OrderedDict(self.encoder))

        # Decoder block
        self.decoder = []
        for i in range(1, len(decoder_structure) - 1):
            self.decoder.append((f'fc{i+1}', nn.Linear(decoder_structure[i-1], decoder_structure[i])))
            self.decoder.append((f'relu{i+1}', nn.LeakyReLU(0.2)))
        self.decoder.append(('fc_outp', nn.Linear(decoder_structure[-2], decoder_structure[-1])))
        self.decoder = nn.Sequential(OrderedDict(self.decoder))

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec


class ScoreATT(nn.Module):
    def __init__(self, cfg) -> None:
        super(ScoreATT, self).__init__()
        latent_dim = cfg.QKV_STRUCTURE
        latent_dim = [6] + latent_dim
        outp_dim = [latent_dim[-1]] + cfg.OUTP_CHN

        # Attention block
        q_structure = []
        k_structure = []
        v_structure = []
        for i in range(1, len(latent_dim)):
            if cfg.USE_BN:
                q_structure += [(f'q_bn{i}', nn.BatchNorm1d(latent_dim[i-1]))]
                k_structure += [(f'k_bn{i}', nn.BatchNorm1d(latent_dim[i-1]))]
                v_structure += [(f'v_bn{i}', nn.BatchNorm1d(latent_dim[i-1]))]
            q_structure += [(f'q_{i}', nn.Conv1d(latent_dim[i-1], latent_dim[i], 1)),
                            (f'q_relu_{i}', nn.LeakyReLU(0.2))]
            k_structure += [(f'k_{i}', nn.Conv1d(latent_dim[i-1], latent_dim[i], 1)),
                            (f'k_relu_{i}', nn.LeakyReLU(0.2))]
            v_structure += [(f'v_{i}', nn.Conv1d(latent_dim[i-1], latent_dim[i], 1)),
                            (f'v_relu_{i}', nn.LeakyReLU(0.2))]
        self.q = nn.Sequential(OrderedDict(q_structure))
        self.k = nn.Sequential(OrderedDict(k_structure))
        self.v = nn.Sequential(OrderedDict(v_structure))

        self.attention = nn.MultiheadAttention(latent_dim[-1], num_heads=cfg.NHEAD)
        self.aggregation = nn.MaxPool1d((23))

        # Output block
        outp_structure = []
        for i in range(1, len(outp_dim)):
            if cfg.USE_BN:
                outp_structure += [(f'outp_bn_{i}', nn.BatchNorm1d(outp_dim[i-1]))]
            outp_structure.append((f'outp{i}', nn.Conv1d(outp_dim[i-1], outp_dim[i], 1)))
        self.outp = nn.Sequential(OrderedDict(outp_structure))

    def forward(self, x):
        x = x.transpose(1, 2)
        # q, k, v as in the attention layer
        q = self.q(x).permute(2, 0, 1)
        k = self.k(x).permute(2, 0, 1)
        v = self.v(x).permute(2, 0, 1)
        
        # Multi-head attention
        att, _ = self.attention(q, k, v)
        att = att.permute(1, 2, 0)

        # Average and concatenation
        aggr_att = self.aggregation(att)

        # 1x1 conv and average again
        outp = self.outp(aggr_att).squeeze(1)

        return outp
