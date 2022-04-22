import torch
import torch.nn as nn
from collections import OrderedDict


class PredTD(nn.Module):
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
    def __init__(self, cfg):
        super(PredAE, self).__init__()
        latent_dim = cfg.LAT_D
        feature_dim = cfg.IN_DIM
        encoder_structure = cfg.STRUCTURE
        encoder_structure = [feature_dim] + encoder_structure + [latent_dim]
        decoder_structure = list(reversed(encoder_structure))

        self.encoder = []
        for i in range(1, len(encoder_structure) - 1):
            self.encoder.append((f'fc{i+1}', nn.Linear(encoder_structure[i-1], encoder_structure[i])))
            self.encoder.append((f'relu{i+1}', nn.LeakyReLU(0.2)))
        self.encoder.append(('fc_outp', nn.Linear(encoder_structure[-2], encoder_structure[-1])))
        self.encoder = nn.Sequential(OrderedDict(self.encoder))

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

class PredATT(nn.Module):
    def __init__(self, cfg) -> None:
        super(PredATT, self).__init__()
        latent_dim = cfg.QKV_STRUCTURE
        self.attention = nn.MultiheadAttention(latent_dim[-1], num_heads=cfg.NHEAD)
        latent_dim = [6] + latent_dim
        outp_dim = [latent_dim[-1] * 2] + cfg.OUTP_CHN
        if cfg.USE_BN:
            q_structure = []
            k_structure = []
            v_structure = []
            for i in range(1, len(latent_dim)):
                q_structure += [(f'q_bn{i}', nn.BatchNorm1d(23)),
                                (f'q_{i}', nn.Linear(latent_dim[i-1], latent_dim[i])),
                                (f'q_relu_{i}', nn.LeakyReLU(0.2))]
                k_structure += [(f'k_bn{i}', nn.BatchNorm1d(23)),
                                (f'k_{i}', nn.Linear(latent_dim[i-1], latent_dim[i])),
                                (f'k_relu_{i}', nn.LeakyReLU(0.2))]
                v_structure += [(f'v_bn{i}', nn.BatchNorm1d(23)),
                                (f'v_{i}', nn.Linear(latent_dim[i-1], latent_dim[i])),
                                (f'v_relu_{i}', nn.LeakyReLU(0.2))]
            self.q = nn.Sequential(OrderedDict(q_structure))
            self.k = nn.Sequential(OrderedDict(k_structure))
            self.v = nn.Sequential(OrderedDict(v_structure))
        else:
            q_structure = []
            k_structure = []
            v_structure = []
            for i in range(1, len(latent_dim)):
                q_structure += [(f'q_{i}', nn.Linear(latent_dim[i-1], latent_dim[i])),
                                (f'q_relu_{i}', nn.LeakyReLU(0.2))]
                k_structure += [(f'k_{i}', nn.Linear(latent_dim[i-1], latent_dim[i])),
                                (f'k_relu_{i}', nn.LeakyReLU(0.2))]
                v_structure += [(f'v_{i}', nn.Linear(latent_dim[i-1], latent_dim[i])),
                                (f'v_relu_{i}', nn.LeakyReLU(0.2))]
            self.q = nn.Sequential(OrderedDict(q_structure))
            self.k = nn.Sequential(OrderedDict(k_structure))
            self.v = nn.Sequential(OrderedDict(v_structure))
        outp_structure = [(f'outp{i}', nn.Conv1d(outp_dim[i-1], outp_dim[i], 1)) for i in range(1, len(outp_dim))]
        outp_structure.append(('flatten', nn.Flatten(1, -1)))
        outp_structure.append(('fc_final', nn.Linear(23, 1)))
        self.outp = nn.Sequential(OrderedDict(outp_structure))

    def forward(self, x):
        # q, k, v as in the attention layer
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        
        # Multi-head attention
        att, _ = self.attention(q, k, v)

        # Average and concatenation
        att_mean, _ = torch.max(att, 1, keepdim=True)
        att_mean = torch.Tensor.repeat(att_mean, (1, att.shape[1], 1))
        att = torch.cat([att, att_mean], -1).transpose(1, 2)

        # 1x1 conv and average again
        outp = self.outp(att)

        return outp
