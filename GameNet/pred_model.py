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
    '''
    Self-attention network to predict score label
    '''
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


class GameMLP(nn.Module):
    '''
    Basic MLP model for decision making
    '''
    def __init__(self, cfg) -> None:
        super(GameMLP, self).__init__()
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
        self.net = nn.Sequential(OrderedDict(self.net))

    def forward(self, x):
        return self.net(x)


class GATLayer(nn.Module):
    def __init__(self, cfg, inp_dim=7) -> None:
        super(GATLayer, self).__init__()
        
        # Message block
        message_structure = [inp_dim] + cfg.MSG.STRUCTURE
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
        self.attention = nn.Linear(message_structure[-1] * 2, cfg.AGG.NHEAD)
        self.normalize = nn.Softmax(2)
        if cfg.AGG.NHEAD > 1:
            self.squeeze_back = nn.Linear(cfg.AGG.NHEAD * message_structure[-1], message_structure[-1])
        

    def forward(self, x):
        # Extract vertice features
        fx = self.message(x)

        # Do pair-fusion
        fx_cat = torch.zeros(x.shape[0], x.shape[1], x.shape[1], fx.shape[-1] * 2)
        for i in range(fx_cat.shape[1]):
            fx_cat[:, i, :, :fx.shape[-1]] = fx[:, i].unsqueeze(1)
        for i in range(fx_cat.shape[2]):
            fx_cat[:, :, i, fx.shape[-1]:] = fx[:, i].unsqueeze(1)

        # Calculate attention weights
        att = self.attention(fx_cat)
        att = self.normalize(att)       # bs x 23 x 23 x nhead

        att_cat = []
        for i in range(att.shape[-1]):
            att_item = torch.zeros_like(fx)
            for j in range(x.shape[1]):
                att_item[:, j, :] = torch.sum(torch.mul(att[:, j, :, i:i+1], fx), 1)
            att_cat.append(att_item)
        
        att_cat = torch.cat(att_cat, -1)

        if att.shape[-1] > 1:
            att_cat = self.squeeze_back(att_cat)
        
        return att_cat


class GameGAT(nn.Module):
    def __init__(self, cfg) -> None:
        super(GameGAT, self).__init__()

        self.gat1 = GATLayer(cfg)
        self.gat2 = GATLayer(cfg, inp_dim=8)

        self.outp = nn.Linear(8, 2)

    def forward(self, x):
        x = self.gat1(x)
        x = self.gat2(x)
        x = torch.mean(x, 1)
        return self.outp(x)


class gameGAT(nn.Module):
    '''
    Graph attention network for decision making
    '''
    def __init__(self, cfg) -> None:
        super(gameGAT, self).__init__()

        # Message passing block
        message_structure = [7] + cfg.MSG.STRUCTURE
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
        for i in range(1, len(aggr_structure)):
            if cfg.AGG.USE_BN:
                att_q += [(f'q_bn_{i}', nn.BatchNorm1d(aggr_structure[i-1]))]
                att_k += [(f'k_bn_{i}', nn.BatchNorm1d(aggr_structure[i-1]))]
                att_v += [(f'v_bn_{i}', nn.BatchNorm1d(aggr_structure[i-1]))]
            att_q += [(f'q_{i}', nn.Conv1d(aggr_structure[i-1], aggr_structure[i], 1)),
                      (f'q_relu_{i}', nn.LeakyReLU(0.2))]
            att_k += [(f'k_{i}', nn.Conv1d(aggr_structure[i-1], aggr_structure[i], 1)),
                      (f'k_relu_{i}', nn.LeakyReLU(0.2))]
            att_v += [(f'v_{i}', nn.Conv1d(aggr_structure[i-1], aggr_structure[i], 1)),
                      (f'v_relu_{i}', nn.LeakyReLU(0.2))]
        self.att_q = nn.Sequential(OrderedDict(att_q))
        self.att_k = nn.Sequential(OrderedDict(att_k))
        self.att_v = nn.Sequential(OrderedDict(att_v))

        self.attention = nn.MultiheadAttention(aggr_structure[-1], num_heads=cfg.AGG.NHEAD)
        self.aggregation = eval(f'nn.{cfg.AGG.POOLING}Pool1d((23))')

        # Output block
        outp_structure = [aggr_structure[-1]] + cfg.OUTP.STRUCTURE
        outp = []
        for i in range(1, len(outp_structure)):
            if cfg.OUTP.USE_BN:
                outp += [(f'outp_bn_{i}', nn.BatchNorm1d(outp_structure[i-1]))]
            outp += [(f'outp_conv_{i}', nn.Conv1d(outp_structure[i-1], outp_structure[i], 1))]
            if i != len(outp_structure) - 1:
                outp += [(f'outp_relu_{i}', nn.LeakyReLU(0.2))]

        self.outp = nn.Sequential(OrderedDict(outp))

    def forward(self, x):
        # Shape of x: bs * 23 * 7 (x, y)
        message = self.message(x)
        summed_msg = torch.sum(message, 1, keepdim=True)
        
        summed_msg = torch.Tensor.repeat(summed_msg, (1, message.shape[1], 1)) - message
        summed_msg = summed_msg.transpose(1, 2)

        q = self.att_q(summed_msg).permute(2, 0, 1)
        k = self.att_k(summed_msg).permute(2, 0, 1)
        v = self.att_v(summed_msg).permute(2, 0, 1)

        att, _ = self.attention(q, k, v)
        att = att.permute(1, 2, 0)
        aggr_att = self.aggregation(att)

        outp = self.outp(aggr_att).squeeze(2)

        return outp
