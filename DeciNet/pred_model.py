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


class PredATT(nn.Module):
    def __init__(self, cfg) -> None:
        super(PredATT, self).__init__()
        latent_dim = cfg.QKV_STRUCTURE
        latent_dim = [6] + latent_dim
        outp_dim = [latent_dim[-1] * 2] + cfg.OUTP_CHN

        # QKV block
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

        # Attention block
        self.attention = nn.MultiheadAttention(latent_dim[-1], num_heads=cfg.NHEAD)

        # Output block
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
        aggr_att, _ = torch.max(att, 1, keepdim=True)
        aggr_att = torch.Tensor.repeat(aggr_att, (1, att.shape[1], 1))
        att = torch.cat([att, aggr_att], -1).transpose(1, 2)

        # 1x1 conv and average again
        outp = self.outp(att)

        return outp


class PredGAT(nn.Module):
    def __init__(self, cfg) -> None:
        super(PredGAT, self).__init__()

        # Message passing block
        message_structure = [2] + cfg.MSG.STRUCTURE
        self.message = []
        for i in range(1, len(message_structure) - 1):
            if cfg.MSG.USE_BN:
                self.message.append((f'msg_bn_{i}', nn.BatchNorm1d(23)))
            self.message.append((f'msg_fc_{i}', nn.Linear(message_structure[i - 1], message_structure[i])))
            self.message.append((f'msg_relu_{i}', nn.LeakyReLU(0.2)))
        if cfg.MSG.USE_BN:
            self.message.append((f'msg_bn_{len(message_structure) - 1}', nn.BatchNorm1d(23)))
        self.message.append((f'msg_fc_{len(message_structure) - 1}', nn.Linear(message_structure[-2], message_structure[-1])))
        self.message = nn.Sequential(OrderedDict(self.message))

        # Attention block
        aggr_structure = [message_structure[-1]] + cfg.AGG.STRUCTURE
        att_q = []
        att_k = []
        att_v = []
        if cfg.AGG.USE_BN:
            for i in range(1, len(aggr_structure)):
                att_q += [(f'q_bn_{i}', nn.BatchNorm1d(23)),
                          (f'q_{i}', nn.Linear(aggr_structure[i-1], aggr_structure[i])),
                          (f'q_relu_{i}', nn.LeakyReLU(0.2))]
                att_k += [(f'k_bn_{i}', nn.BatchNorm1d(23)),
                          (f'k_{i}', nn.Linear(aggr_structure[i-1], aggr_structure[i])),
                          (f'k_relu_{i}', nn.LeakyReLU(0.2))]
                att_v += [(f'v_bn_{i}', nn.BatchNorm1d(23)),
                          (f'v_{i}', nn.Linear(aggr_structure[i-1], aggr_structure[i])),
                          (f'v_relu_{i}', nn.LeakyReLU(0.2))]
            self.att_q = nn.Sequential(OrderedDict(att_q))
            self.att_k = nn.Sequential(OrderedDict(att_k))
            self.att_v = nn.Sequential(OrderedDict(att_v))
        else:
            for i in range(1, len(aggr_structure)):
                att_q += [(f'q_{i}', nn.Linear(aggr_structure[i-1], aggr_structure[i])),
                          (f'q_relu_{i}', nn.LeakyReLU(0.2))]
                att_k += [(f'k_{i}', nn.Linear(aggr_structure[i-1], aggr_structure[i])),
                          (f'k_relu_{i}', nn.LeakyReLU(0.2))]
                att_v += [(f'v_{i}', nn.Linear(aggr_structure[i-1], aggr_structure[i])),
                          (f'v_relu_{i}', nn.LeakyReLU(0.2))]
            self.att_q = nn.Sequential(OrderedDict(att_q))
            self.att_k = nn.Sequential(OrderedDict(att_k))
            self.att_v = nn.Sequential(OrderedDict(att_v))
        
        self.attention = nn.MultiheadAttention(aggr_structure[i], num_heads=cfg.AGG.NHEAD)
        self.aggregation = eval(f'nn.{cfg.AGG.POOLING}Pool2d((23, 1))')

        # Output block
        outp_structure = [aggr_structure[-1] * 2] + cfg.OUTP.STRUCTURE
        outp = []
        for i in range(1, len(outp_structure)):
            outp += [(f'outp_conv_{i}', nn.Conv1d(outp_structure[i-1], outp_structure[i], 1)),
                     (f'outp_relu_{i}', nn.ReLU(0.2))]
        self.outp1 = nn.Sequential(OrderedDict(outp))
        self.outp2 = nn.Linear(outp_structure[-1], 2)
        if cfg.OUTP.USE_BN:
            self.outp2 = nn.Sequential(OrderedDict([('outp_bnf', nn.BatchNorm1d(23)), ('outp_fc', self.outp2)]))

    def forward(self, x):
        # Shape of x: bs * 23 * 3 (x, y, team_id)
        message = self.message(x)
        summed_msg = torch.mean(message, 1, keepdim=True)
        
        summed_msg = torch.Tensor.repeat(summed_msg, (1, message.shape[1], 1)) - message

        q = self.att_q(summed_msg)
        k = self.att_k(summed_msg)
        v = self.att_v(summed_msg)

        att, _ = self.attention(q, k, v)
        aggr_att = self.aggregation(att)
        aggr_att = torch.Tensor.repeat(aggr_att, (1, att.shape[1], 1))
        aggr_att = torch.cat([att, aggr_att], -1).transpose(1, 2)

        outp = self.outp1(aggr_att)
        outp = outp.transpose(1, 2)
        outp = self.outp2(outp)

        return self.crop_decision(outp)

    def crop_decision(self, out):
        pass


class OffenseGAT(PredGAT):
    def __init__(self, cfg) -> None:
        super(OffenseGAT, self).__init__(cfg)
    def crop_decision(self, out):
        return out[:, :11, :]

class DefenseGAT(PredGAT):
    def __init__(self, cfg) -> None:
        super(DefenseGAT, self).__init__(cfg)
    def crop_decision(self, out):
        return out[:, 11:22, :]
