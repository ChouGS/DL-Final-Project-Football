import os

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

def make_off_data(data, pos, decision):
    '''
        data: [bs, 23, 2], positions of the players+ball
        pos: [bs, 2], position of each current offensive player
        decision: [bs, 11, 2], decision of each player
    '''

    import pdb
    pdb.set_trace()
    pass