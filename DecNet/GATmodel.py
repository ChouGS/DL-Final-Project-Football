import torch
import torch.nn as nn
from collections import OrderedDict

class PredGAT(nn.Module):
    def __init__(self, cfg) -> None:
        super(PredGAT, self).__init__()

        # Message passing block
        message_structure = [3] + cfg.MSG.STRUCTURE
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
        self.aggregation = eval(f'nn.{cfg.AGG.POOLING}Pooling2d(23, 1)')

        # Output block
        outp_structure = [aggr_structure[-1] * 2] + cfg.OUTP.STRUCTURE
        outp = []
        for i in range(1, len(outp_structure)):
            outp += [(f'outp_conv_{i}', nn.Conv1d(outp_structure[i-1], outp_structure[i], 1)),
                     (f'outp_relu_{i}', nn.ReLU(0.2))]
        self.outp1 = nn.Sequential(OrderedDict(outp))
        self.outp2 = nn.Linear(outp_structure[-1], 2)

    def forward(self, x):
        # Shape of x: bs * 23 * 3 (x, y, team_id)
        message = self.message(x)
        summed_msg, _ = torch.max(message, 0, keepdim=True)
        summed_msg = torch.Tensor.repeat(summed_msg, (message.shape[0], 1)) - message

        q = self.att_q(summed_msg)
        k = self.att_k(summed_msg)
        v = self.att_v(summed_msg)

        att, _ = self.attention(q, k, v)
        aggr_att = self.aggregation(att)
        aggr_att = torch.Tensor.repeat(aggr_att, (1, aggr_att.shape[1], 1))
        aggr_att = torch.cat([att, aggr_att], -1).transpose(1, 2)

        outp = self.outp1(aggr_att)
        outp = outp.transpose(1, 2)
        outp = self.outp2(outp)

        return outp
