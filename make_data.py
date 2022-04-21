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
    for fname in os.listdir(os.path.join(data_root, 'before_passing')):
        data_item = np.load(os.path.join(data_root, 'before_passing', fname))
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
    import pdb
    pdb.set_trace()
    np.save(os.path.join(data_root, 'data_after_passing.npy'), data)

    data = np.load(os.path.join(data_root, 'data_before_passing.npy'))
    touchdown = (np.abs(data[:, -1] - 400) < 1).astype(data.dtype)[:, np.newaxis]
    data = np.concatenate([data, touchdown], 1)
    np.save(os.path.join(data_root, 'data_before_passing.npy'), data)

def append_ball_dist(data_root):
    data = np.load(os.path.join(data_root, 'data_after_passing.npy'))
    ball_pos = data[:, 63:65]
    self_pos = data[:, 65:67]
    ball_dist = np.sqrt(np.sum((ball_pos - self_pos) * (ball_pos - self_pos), 1, keepdims=True))
    data = np.concatenate([data[:, :65], ball_dist, data[:, 65:]], 1)
    np.save(os.path.join(data_root, 'data_after_passing.npy'), data)

    data = np.load(os.path.join(data_root, 'data_before_passing.npy'))
    ball_pos = data[:, 63:65]
    self_pos = data[:, 65:67]
    ball_dist = np.sqrt(np.sum((ball_pos - self_pos) * (ball_pos - self_pos), 1, keepdims=True))
    
    data = np.concatenate([data[:, :65], ball_dist, data[:, 65:]], 1)
    np.save(os.path.join(data_root, 'data_before_passing.npy'), data)

def append_self_pos(data_root, nplayers=11):
    data = np.load(os.path.join(data_root, 'data_after_passing.npy'))
    assert data.shape[1] == (2 * nplayers - 1) * 3 + 11
    assert data.shape[0] % (2 * nplayers) == 0
    new_data = np.zeros((data.shape[0], 3 * (2 * nplayers) + 11))
    for i in range(data.shape[0]):
        new_data[i] = np.insert(data[i], 3 * (i % (2 * nplayers)), [data[i, 66], data[i, 67], 0])
    np.save(os.path.join(data_root, 'data_after_passing_p.npy'), new_data)

    data = np.load(os.path.join(data_root, 'data_before_passing.npy'))
    assert data.shape[1] == (2 * nplayers - 1) * 3 + 11
    assert data.shape[0] % (2 * nplayers) == 0
    new_data = np.zeros((data.shape[0], 3 * (2 * nplayers) + 11))
    for i in range(data.shape[0]):
        new_data[i] = np.insert(data[i], 3 * (i % (2 * nplayers)), [data[i, 66], data[i, 67], 0])
    np.save(os.path.join(data_root, 'data_before_passing_p.npy'), new_data)

def modify_ball_dist(root_dir, nplayers=11):
    data = np.load(os.path.join(data_root, 'data_after_passing.npy'))
    for i in range(data.shape[0]):
        if i % (2 * nplayers) < 11 and data[i, 68] > 0:
            data[i, 68] = -data[i, 68]
        if i % (2 * nplayers) >= 11 and data[i, 68] < 0:
            data[i, 68] = -data[i, 68]
    np.save(os.path.join(data_root, 'data_after_passing.npy'), data)

    data = np.load(os.path.join(data_root, 'data_before_passing.npy'))
    for i in range(data.shape[0]):
        if i % (2 * nplayers) < 11 and data[i, 68] > 0:
            data[i, 68] = -data[i, 68]
        if i % (2 * nplayers) >= 11 and data[i, 68] < 0:
            data[i, 68] = -data[i, 68]
    np.save(os.path.join(data_root, 'data_before_passing.npy'), data)


if __name__ == '__main__':
    data_root = 'raw_data/11oLpHcL/synthesized'
    # make_data(data_root)
    # append_ball_dist(data_root)
    append_touchdown_label(data_root)
    # append_self_pos(data_root)
    

