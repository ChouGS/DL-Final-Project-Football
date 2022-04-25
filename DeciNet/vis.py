import numpy as np
import os
from matplotlib import pyplot as plot

def vis_loss_curve(cfg, loss_dict):
    os.makedirs(f'DeciNet/results/{cfg.NAME}/loss', exist_ok=True)
    for i, key in enumerate(loss_dict.keys()):
        plot.figure(i)
        plot.plot(np.arange(len(loss_dict[key])), np.array(loss_dict[key]), 'g-')
        plot.xlabel('iters')
        plot.ylabel('loss')
        plot.title(f'{key}')
        plot.savefig(f'DeciNet/results/{cfg.NAME}/loss/{key}.jpg')
