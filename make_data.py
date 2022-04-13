import numpy as np
import os

root_list = ['cls_dataset_1tick', '1', '2', '3']

data_ap = None
for root in root_list:
    for fname in os.listdir(os.path.join(root, 'after_passing')):
        data_item = np.load(os.path.join(root, 'after_passing', fname))
        if data_ap is None:
            data_ap = data_item
        else:
            data_ap = np.concatenate([data_ap, data_item], 0)

data_bp = None
for root in root_list:
    for fname in os.listdir(os.path.join(root, 'before_passing')):
        data_item = np.load(os.path.join(root, 'before_passing', fname))
        if data_bp is None:
            data_bp = data_item
        else:
            data_bp = np.concatenate([data_bp, data_item], 0)

print(data_ap.shape)
print(data_bp.shape)
np.save('data_after_passing.npy', data_ap)
np.save('data_before_passing.npy', data_bp)
