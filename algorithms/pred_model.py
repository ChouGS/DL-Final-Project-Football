import torch
import torch.nn as nn
from collections import OrderedDict


class GameGAT(nn.Module):
    '''
    Graph attention network for decision making
    '''
    def __init__(self, cfg) -> None:
        super(GameGAT, self).__init__()

        # Message passing block
        message_structure = [7] + cfg.MSG.STRUCTURE
        self.message = []
        for i in range(1, len(message_structure) - 1):
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
        # Shape of x: bs * 23 * 2 (x, y)
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
