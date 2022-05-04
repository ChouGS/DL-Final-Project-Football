import numpy as np
import os
from matplotlib import pyplot as plot

def vis_loss_curve(cfg, loss_dict, name, samp_rate=5):
    os.makedirs(f'DeciNet/results/{cfg.NAME}/loss', exist_ok=True)
    for i, key in enumerate(loss_dict.keys()):
        plot.figure(i)
        plot.plot(np.arange(0, len(loss_dict[key]), samp_rate), np.array(loss_dict[key][::5]), 'g-')
        plot.xlabel('iters')
        plot.ylabel('loss')
        plot.title(f'{key}')
        plot.savefig(f'DeciNet/results/{cfg.NAME}/loss/{name}_{key}.jpg')
