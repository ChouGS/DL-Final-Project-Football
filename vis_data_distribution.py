import numpy as np
from matplotlib import pyplot as plot

def vis_distribution():
    ap_data = np.load('data_after_passing.npy')
    bp_data = np.load('data_before_passing.npy')

    plot.hist(ap_data[:, -1], bins=80, range=(ap_data[:, -1].min(), ap_data[:, -1].max()), facecolor='green', edgecolor='black')
    plot.xlabel('label range')
    plot.ylabel('label frequency')
    plot.title('After passing data distribution')
    plot.savefig('ap_fig_dist.jpg')
    plot.cla()

    plot.hist(bp_data[:, -1], bins=80, range=(bp_data[:, -1].min(), bp_data[:, -1].max()), facecolor='green', edgecolor='black')
    plot.xlabel('label range')
    plot.ylabel('label frequency')
    plot.title('After passing data distribution')
    plot.savefig('bp_fig_dist.jpg')
    plot.cla()
