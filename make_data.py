import numpy as np
import os

def make_data(data_root):
    data_ap = None
    for fname in os.listdir(os.path.join('raw_data', data_root, 'after_passing')):
        data_item = np.load(os.path.join('raw_data', data_root, 'after_passing', fname))
        if data_ap is None:
            data_ap = data_item
        else:
            data_ap = np.concatenate([data_ap, data_item], 0)

    data_bp = None
    for fname in os.listdir(os.path.join('raw_data', data_root, 'before_passing')):
        data_item = np.load(os.path.join('raw_data', data_root, 'before_passing', fname))
        if data_bp is None:
            data_bp = data_item
        else:
            data_bp = np.concatenate([data_bp, data_item], 0)

    print(data_ap.shape)
    print(data_bp.shape)
    os.makedirs(os.path.join('raw_data', data_root, 'synthesized'), exist_ok=True)
    np.save(os.path.join('raw_data', data_root, 'synthesized', 'data_after_passing.npy'), data_ap)
    np.save(os.path.join('raw_data', data_root, 'synthesized', 'data_before_passing.npy'), data_bp)

def append_touchdown_label(data_root):
    data = np.load(os.path.join(data_root, 'data_after_passing.npy'))
    touchdown = (np.abs(data[:, -1] - 400) < 1).astype(data.dtype)[:, np.newaxis]
    data = np.concatenate([data, touchdown], 1)
    np.save(os.path.join(data_root, 'data_after_passing.npy'), data)

    data = np.load(os.path.join(data_root, 'data_before_passing.npy'))
    touchdown = (np.abs(data[:, -1] - 400) < 1).astype(data.dtype)[:, np.newaxis]
    data = np.concatenate([data, touchdown], 1)
    np.save(os.path.join(data_root, 'data_before_passing.npy'), data)

if __name__ == '__main__':
    data_root = 'raw_data/11oLpHcH/synthesized'
    append_touchdown_label(data_root)
