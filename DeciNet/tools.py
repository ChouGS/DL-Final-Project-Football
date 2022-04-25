import os
import torch
from torch import isin

def find_last_epoch(root):
    epoch = 0
    res_fname = None
    if os.path.exists(root):
        for fname in os.listdir(root):
            cur_epoch = fname.split('.')[0].split('_')[-1]
            if cur_epoch.isdigit():
                if int(cur_epoch) > epoch:
                    epoch = int(cur_epoch)
                    res_fname = os.path.join(root, fname)
            elif cur_epoch == 'final':
                epoch = -1
                res_fname = os.path.join(root, fname)
                break
    return epoch, res_fname

def make_pred_data(data, position, decision):
    '''
        data: [bs, 23, 2], positions of the players+ball
        position: [bs, 2], position of the player
        decision: [bs, 2], decision of the player
    '''
    nplayers = data.shape[1]

    position = position.unsqueeze(1)
    position = torch.Tensor.repeat(position, (1, nplayers, 1))

    decision = decision.unsqueeze(1)
    decision = torch.Tensor.repeat(decision, (1, nplayers, 1))

    off_data = torch.cat([data, position, decision], 2)

    return off_data

def make_gat_data(data, position, v):
    '''
        data: [bs, 23, 2], positions of the players+ball
        position: [bs, 2], position of the player
        decision: [bs, 2], decision of the player
    '''
    nplayers = data.shape[1]

    position = position.unsqueeze(1)
    position = torch.Tensor.repeat(position, (1, nplayers, 1))

    v = v.unsqueeze(1)
    v = torch.Tensor.repeat(v, (1, nplayers, 1))

    v_max = torch.ones(data.shape[0], data.shape[1], 1) * 10

    off_data = torch.cat([data, position, v, v_max], 2)

    return off_data
