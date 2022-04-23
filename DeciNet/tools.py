import os
import torch
from torch import isin

def find_last_epoch(root):
    epoch = 0
    res_fname = None
    if os.path.exists(root):
        for fname in os.listdir(root):
            cur_epoch = fname.split('.')[0].split('/')[-1]
            if cur_epoch.isdigit():
                if int(cur_epoch) > epoch:
                    epoch = int(cur_epoch)
                    res_fname = os.path.join(root, fname)
    return epoch, res_fname

def make_off_data(data, decision):
    '''
        data: [bs, 23, 2], positions of the players+ball
        decision: [bs, 11, 2], decision of each offensive player
    '''
    bsize = data.shape[0]
    nplayers = decision.shape[1]

    off_data = data.unsqueeze(1)
    off_data = torch.Tensor.repeat(off_data, (1, nplayers, 1, 1))
    off_data = torch.flatten(off_data, 0, 1)

    off_pos = data[:, :11, :]
    off_pos = off_pos.unsqueeze(1)
    off_pos = torch.Tensor.repeat(off_pos, (1, 2 * nplayers + 1, 1, 1)).transpose(1, 2)
    off_pos = torch.flatten(off_pos, 0, 1)

    decision = decision.unsqueeze(1)
    decision = torch.Tensor.repeat(decision, (1, 2 * nplayers + 1, 1, 1)).transpose(1, 2)
    decision = torch.flatten(decision, 0, 1)

    off_data = torch.cat((off_data, off_pos, decision), 2)

    return off_data


def make_def_data(data, decision):
    '''
        data: [bs, 23, 2], positions of the players+ball
        decision: [bs, 11, 2], decision of each defensive player
    '''
    bsize = data.shape[0]
    nplayers = decision.shape[1]

    def_data = data.unsqueeze(1)
    def_data = torch.Tensor.repeat(def_data, (1, nplayers, 1, 1))
    def_data = torch.flatten(def_data, 0, 1)

    def_pos = data[:, 11:22, :]
    def_pos = def_pos.unsqueeze(1)
    def_pos = torch.Tensor.repeat(def_pos, (1, 2 * nplayers + 1, 1, 1)).transpose(1, 2)
    def_pos = torch.flatten(def_pos, 0, 1)

    decision = decision.unsqueeze(1)
    decision = torch.Tensor.repeat(decision, (1, 2 * nplayers + 1, 1, 1)).transpose(1, 2)
    decision = torch.flatten(decision, 0, 1)

    def_data = torch.cat((def_data, def_pos, decision), 2)

    return def_data
