import numpy as np
import os
from matplotlib import pyplot as plot

def vis_distribution():
    for root in os.listdir('raw_data'):
        if os.path.exists(os.path.join('raw_data', root, 'synthesized')):
            ap_data = np.load(os.path.join('raw_data', root, 'synthesized', 'data_after_passing.npy'))
            bp_data = np.load(os.path.join('raw_data', root, 'synthesized', 'data_before_passing.npy'))

            plot.hist(ap_data[:, -1], bins=80, range=(ap_data[:, -1].min(), ap_data[:, -1].max()), facecolor='green', edgecolor='black')
            plot.xlabel('label range')
            plot.ylabel('label frequency')
            plot.title('After passing data distribution')
            plot.savefig(os.path.join('raw_data', root, 'synthesized', 'ap_fig_dist.jpg'))
            plot.cla()

            plot.hist(bp_data[:, -1], bins=80, range=(bp_data[:, -1].min(), bp_data[:, -1].max()), facecolor='green', edgecolor='black')
            plot.xlabel('label range')
            plot.ylabel('label frequency')
            plot.title('After passing data distribution')
            plot.savefig(os.path.join('raw_data', root, 'synthesized', 'bp_fig_dist.jpg'))
            plot.cla()
