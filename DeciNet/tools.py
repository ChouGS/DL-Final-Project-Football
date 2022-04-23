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
