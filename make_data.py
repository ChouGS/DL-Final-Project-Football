import numpy as np
from matplotlib import pyplot as plot
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

def make_scored_data(root_list, step=10, nplayer=11, beta=0.8):
    ball_x_index = 63
    
    data_ap = None
    for root in root_list:
        d = os.path.join('raw_data', root, 'after_passing')
        for fname in os.listdir(d):
            # Load basic data (72)
            data = np.load(os.path.join(d, fname))
            
            # Append score (72->73)
            data = np.concatenate([data, np.zeros((data.shape[0], 1))], 1)
            beta_series = np.array([beta ** i for i in range(step)])
            ball_x = data[list(range(0, data.shape[0], 2 * nplayer)), ball_x_index] 
            n_valid_ticks = ball_x.shape[0]
            if np.abs(data[0, -2] - 400) < 1:
                ball_x = np.concatenate([ball_x, [410 + 10 * i for i in range(step)]], 0)
            else:
                ball_x = np.concatenate([ball_x, [data[0, -2] for _ in range(step)]], 0)
            for i in range(n_valid_ticks):
                ball_x_diff = ball_x[i + 1:i + step + 1] - ball_x[i:i + step]
                tick_scores = np.dot(beta_series, ball_x_diff)
                data[i * (2 * nplayer):(i + 1) * (2 * nplayer), -1] = tick_scores
                if i == 1:
                    data[0:22, -1] = tick_scores
            
            # Append ball_dist (73->74)
            ball_pos = data[:, 63:65]
            self_pos = data[:, 65:67]
            ball_dist = np.sqrt(np.sum((ball_pos - self_pos) * (ball_pos - self_pos), 1, keepdims=True))
            offender_index = np.arange(data.shape[0]).reshape((-1, 2 * nplayer))[:, :nplayer].reshape((-1))
            ball_dist[offender_index] *= -1
            data = np.concatenate([data[:, :65], ball_dist, data[:, 65:]], 1)

            # Append touchdown label (74->75)
            touchdown = (np.abs(data[:, -2] - 400) < 1).astype(data.dtype)[:, np.newaxis]
            data = np.concatenate([data, touchdown], 1)

            # Append self position (75->78)
            new_data = np.zeros((data.shape[0], 3 * (2 * nplayer) + 12))
            for i in range(data.shape[0]):
                new_data[i] = np.insert(data[i], 3 * (i % (2 * nplayer)), [data[i, 66], data[i, 67], 0])
            data = new_data

            # Append last velocity (78->80)
            prev_v = np.zeros((data.shape[0], 2))
            for i in range(2 * nplayer):
                prev_v[i] = data[i, 73:75]
                prev_v[i + 2 * nplayer:data.shape[0]:2 * nplayer] = data[i:data.shape[0] - 2 * nplayer:2 * nplayer, 73:75]
            data = np.concatenate([data[:, :71], prev_v, data[:, 71:]], 1)

            if data_ap is None:
                data_ap = data
            else:
                data_ap = np.concatenate([data_ap, data], 0)
        
    data_bp = None
    for root in root_list:
        d = os.path.join('raw_data', root, 'before_passing')
        for fname in os.listdir(d):
            # Load basic data (72)
            data = np.load(os.path.join(d, fname))
            
            # Append score (72->73)
            data = np.concatenate([data, np.zeros((data.shape[0], 1))], 1)
            beta_series = np.array([beta ** i for i in range(step)])
            ball_x = data[list(range(0, data.shape[0], 2 * nplayer)), ball_x_index] 
            n_valid_ticks = ball_x.shape[0]
            if np.abs(data[0, -2] - 400) < 1:
                ball_x = np.concatenate([ball_x, [410 + 10 * i for i in range(step)]], 0)
            else:
                ball_x = np.concatenate([ball_x, [data[0, -2] for _ in range(step)]], 0)
            for i in range(n_valid_ticks):
                ball_x_diff = ball_x[i + 1:i + step + 1] - ball_x[i:i + step]
                tick_scores = np.dot(beta_series, ball_x_diff)
                data[i * (2 * nplayer):(i + 1) * (2 * nplayer), -1] = tick_scores
                if i == 1:
                    data[0:22, -1] = tick_scores
            
            # Append ball_dist (73->74)
            ball_pos = data[:, 63:65]
            self_pos = data[:, 65:67]
            ball_dist = np.sqrt(np.sum((ball_pos - self_pos) * (ball_pos - self_pos), 1, keepdims=True))
            offender_index = np.arange(data.shape[0]).reshape((-1, 2 * nplayer))[:, :nplayer].reshape((-1))
            ball_dist[offender_index] *= -1
            data = np.concatenate([data[:, :65], ball_dist, data[:, 65:]], 1)

            # Append touchdown label (74->75)
            touchdown = (np.abs(data[:, -2] - 400) < 1).astype(data.dtype)[:, np.newaxis]
            data = np.concatenate([data, touchdown], 1)

            # Append self position (75->78)
            new_data = np.zeros((data.shape[0], 3 * (2 * nplayer) + 12))
            for i in range(data.shape[0]):
                new_data[i] = np.insert(data[i], 3 * (i % (2 * nplayer)), [data[i, 66], data[i, 67], 0])
            data = new_data

            # Append last velocity (78->80)
            prev_v = np.zeros((data.shape[0], 2))
            for i in range(2 * nplayer):
                prev_v[i] = data[i, 73:75]
                prev_v[i + 2 * nplayer:data.shape[0]:2 * nplayer] = data[i:data.shape[0] - 2 * nplayer:2 * nplayer, 73:75]
            data = np.concatenate([data[:, :71], prev_v, data[:, 71:]], 1)

            if data_bp is None:
                data_bp = data
            else:
                data_bp = np.concatenate([data_bp, data], 0)

    np.save(os.path.join('raw_data', '11oLpHcL', 'synthesized', 'data_after_passing.npy'), data_ap)
    np.save(os.path.join('raw_data', '11oLpHcL', 'synthesized', 'data_before_passing.npy'), data_bp)


def append_touchdown_label(data_root):
    data = np.load(os.path.join(data_root, 'data_after_passing.npy'))
    touchdown = (np.abs(data[:, -2] - 400) < 1).astype(data.dtype)[:, np.newaxis]
    data = np.concatenate([data, touchdown], 1)
    np.save(os.path.join(data_root, 'data_after_passing.npy'), data)

    data = np.load(os.path.join(data_root, 'data_before_passing.npy'))
    touchdown = (np.abs(data[:, -2] - 400) < 1).astype(data.dtype)[:, np.newaxis]
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
    assert data.shape[1] == (2 * nplayers - 1) * 3 + 12
    assert data.shape[0] % (2 * nplayers) == 0
    new_data = np.zeros((data.shape[0], 3 * (2 * nplayers) + 12))
    for i in range(data.shape[0]):
        new_data[i] = np.insert(data[i], 3 * (i % (2 * nplayers)), [data[i, 66], data[i, 67], 0])
    np.save(os.path.join(data_root, 'data_after_passing.npy'), new_data)

    data = np.load(os.path.join(data_root, 'data_before_passing.npy'))
    assert data.shape[1] == (2 * nplayers - 1) * 3 + 12
    assert data.shape[0] % (2 * nplayers) == 0
    new_data = np.zeros((data.shape[0], 3 * (2 * nplayers) + 12))
    for i in range(data.shape[0]):
        new_data[i] = np.insert(data[i], 3 * (i % (2 * nplayers)), [data[i, 66], data[i, 67], 0])
    np.save(os.path.join(data_root, 'data_before_passing.npy'), new_data)

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
    # append_touchdown_label(data_root)
    # append_ball_dist(data_root)
    # for beta in np.arange(0.2, 1, 0.05):
    # append_self_pos(data_root)
    make_scored_data(['1', '2', '3', '11oLpHcL', 'cls_dataset_1tick'], beta=0.9)
