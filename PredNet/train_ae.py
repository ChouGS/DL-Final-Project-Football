from boto import config
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plot
import os
import argparse

from pred_model import PredAE, PredX, PredTD
from dataset import PredictorDataset
from config.default import get_default_cfg

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser()
parser.add_argument("-cfg", "--config", help="[Required] the path to a .yaml file to use as the config.", \
                    type=str, required=True)
args = parser.parse_args()

cfg = get_default_cfg()
cfg.merge_from_file(args.config)

EPOCH = 35
BATCH_SIZE = 256
TEST_BSIZE = 1024
LR = 0.002
TEST_PORTION = 0.2
PRINT_FREQ = 50
DATA_ROOT = 'raw_data/11oLpHcL/cls_dataset_1tick/synthesized'

if __name__ == '__main__':
    # Load data
    ap_data = np.load(os.path.join(DATA_ROOT, 'data_after_passing.npy'))
    bp_data = np.load(os.path.join(DATA_ROOT, 'data_before_passing.npy'))

    ap_train = PredictorDataset(ap_data[:int(round(ap_data.shape[0] * (1 - TEST_PORTION)))])
    bp_train = PredictorDataset(bp_data[:int(round(bp_data.shape[0] * (1 - TEST_PORTION)))])
    ap_tr_notd = PredictorDataset(ap_data[ap_data[:, -1] == 0])
    bp_tr_notd = PredictorDataset(bp_data[bp_data[:, -1] == 0])
    ap_test = PredictorDataset(ap_data[int(round(ap_data.shape[0] * (1 - TEST_PORTION))):])
    bp_test = PredictorDataset(bp_data[int(round(bp_data.shape[0] * (1 - TEST_PORTION))):])

    ap_trloader = DataLoader(ap_train, batch_size=BATCH_SIZE)
    bp_trloader = DataLoader(bp_train, batch_size=BATCH_SIZE)
    ap_tr_notd_loader = DataLoader(ap_tr_notd, batch_size=BATCH_SIZE)    # No-touchdown data
    bp_tr_notd_loader = DataLoader(bp_tr_notd, batch_size=BATCH_SIZE)    # No-touchdown data
    ap_teloader = DataLoader(ap_test, batch_size=TEST_BSIZE)
    bp_teloader = DataLoader(bp_test, batch_size=TEST_BSIZE)

    ### Training autoencoder network
    # Define autoencoder model
    ap_ae = PredAE([128, 256, 256, 64])
    bp_ae = PredAE([128, 256, 256, 64])

    # Reconstruction loss
    rec_loss = nn.MSELoss()

    # Optimizer
    ap_ae_opt = optim.Adam(ap_ae.parameters(), lr=LR)
    bp_ae_opt = optim.Adam(bp_ae.parameters(), lr=LR)

    # Training
    ap_ae_loss = []
    bp_ae_loss = []
    print('\nTraining before_passing autoencoder...')
    niters = len(bp_train) // BATCH_SIZE + 1
    for e in range(EPOCH):
        for i, (data, _, _, _) in enumerate(bp_trloader):
            _, data_rec = bp_ae(data)
            
            loss_val = rec_loss(data_rec, data)
            bp_ae_loss.append(loss_val.item())

            bp_ae_opt.zero_grad()
            loss_val.backward()
            bp_ae_opt.step()

            if i % PRINT_FREQ == 0:
                print(f'AE epoch {e}/{EPOCH}, iter {i + 1}/{niters}: bp_rec_loss={loss_val}')

    print('\nTraining after_passing autoencoder...')
    niters = len(ap_train) // BATCH_SIZE + 1
    for e in range(EPOCH):
        for i, (data, _, _, _) in enumerate(ap_trloader):
            _, data_rec = ap_ae(data)
            
            loss_val = rec_loss(data_rec, data)
            ap_ae_loss.append(loss_val.item())

            ap_ae_opt.zero_grad()
            loss_val.backward()
            ap_ae_opt.step()

            if i % PRINT_FREQ == 0:
                print(f'AE epoch {e}/{EPOCH}, iter {i + 1}/{niters}: ap_rec_loss={loss_val}')

    ### Train touchdown network
    # Define touchdown predictor
    ap_td = PredTD([128, 256, 256, 64])
    bp_td = PredTD([128, 256, 256, 64])

    # Optimizer
    ap_td_opt = optim.Adam(ap_td.parameters(), lr=LR)
    bp_td_opt = optim.Adam(bp_td.parameters(), lr=LR)

    # Loss
    td_bce_loss = nn.BCELoss()

    # Training
    ap_td_loss = []
    bp_td_loss = []
    print('\nTraining before_passing touchdown...')
    niters = len(bp_train) // BATCH_SIZE + 1
    for e in range(EPOCH):
        for i, (data, _, _, td_label) in enumerate(bp_trloader):
            # Prepare one-hot label
            td_label1h = torch.zeros(data.shape[0], 2)
            td_label1h[torch.arange(data.shape[0]).long(), td_label] = 1

            td_logit = bp_td(data)
            
            loss_val = td_bce_loss(td_logit, td_label1h)
            bp_td_loss.append(loss_val.item())

            bp_ae_opt.zero_grad()
            loss_val.backward()
            bp_ae_opt.step()

            if i % PRINT_FREQ == 0:
                print(f'TD epoch {e}/{EPOCH}, iter {i + 1}/{niters}: bp_td_loss={loss_val}')

    print('\nTraining after_passing touchdown...')
    niters = len(ap_train) // BATCH_SIZE + 1
    for e in range(EPOCH):
        for i, (data, _, _, td_label) in enumerate(ap_trloader):
            # Prepare one-hot label
            td_label1h = torch.zeros(data.shape[0], 2)
            td_label1h[torch.arange(data.shape[0]).long(), td_label] = 1

            td_logit = ap_td(data)
            
            loss_val = td_bce_loss(td_logit, td_label1h)
            ap_td_loss.append(loss_val.item())

            bp_ae_opt.zero_grad()
            loss_val.backward()
            bp_ae_opt.step()

            if i % PRINT_FREQ == 0:
                print(f'TD epoch {e}/{EPOCH}, iter {i + 1}/{niters}: ap_td_loss={loss_val}')

    # Using the trained autoencoder to train a predictor (only on no-touchdown model)
    pred_loss = nn.MSELoss()

    bp_ae.eval()
    bp_pred = PredX([64, 256, 256, 64, 16], 32)
    bp_pred_opt = optim.Adam(bp_pred.parameters(), lr=LR)
    
    ap_pred_loss = []
    bp_pred_loss = []
    print('\nTraining before_passing predictor...')
    niters = len(bp_tr_notd) // BATCH_SIZE + 1
    for e in range(EPOCH):
        for i, (data, _, x_label, _) in enumerate(bp_tr_notd_loader):
            data_encoded, _ = bp_ae(data)
            logits = bp_pred(data_encoded)
            
            loss_val = pred_loss(logits, x_label)
            bp_pred_loss.append(loss_val.item())

            bp_pred_opt.zero_grad()
            loss_val.backward()
            bp_pred_opt.step()

            if i % PRINT_FREQ == 0:
                print(f'Pred epoch {e}/{EPOCH}, iter {i + 1}/{niters}: bp_pred_loss={loss_val}')

    ap_ae.eval()
    ap_pred = PredX([64, 256, 256, 64, 16], 32)
    ap_pred_opt = optim.Adam(ap_pred.parameters(), lr=LR)

    print('\nTraining after_passing predictor...')
    niters = len(ap_tr_notd) // BATCH_SIZE + 1
    for e in range(EPOCH):
        for i, (data, _, x_label, _) in enumerate(ap_tr_notd_loader):
            data_encoded, _ = ap_ae(data)
            logits = ap_pred(data_encoded)
            
            loss_val = pred_loss(logits, x_label)
            ap_pred_loss.append(loss_val.item())

            ap_pred_opt.zero_grad()
            loss_val.backward()
            ap_pred_opt.step()

            if i % PRINT_FREQ == 0:
                print(f'Pred epoch {e}/{EPOCH}, iter {i + 1}/{niters}: ap_pred_loss={loss_val}')
    
    # Plot loss curves
    plot.figure(0)
    plot.plot(np.arange(len(ap_ae_loss)), np.array(ap_ae_loss), 'g-')
    plot.xlabel('iters')
    plot.ylabel('loss')
    plot.title('After-passing AE Loss')
    plot.savefig('loss_vis/ap_ae_loss.jpg')

    plot.figure(1)
    plot.plot(np.arange(len(bp_ae_loss)), np.array(bp_ae_loss), 'g-')
    plot.xlabel('iters')
    plot.ylabel('loss')
    plot.title('Before-passing AE Loss')
    plot.savefig('loss_vis/bp_ae_loss.jpg')

    plot.figure(2)
    plot.plot(np.arange(len(ap_td_loss)), np.array(ap_td_loss), 'g-')
    plot.xlabel('iters')
    plot.ylabel('loss')
    plot.title('After-passing touchdown Loss')
    plot.savefig('loss_vis/ap_td_loss.jpg')

    plot.figure(3)
    plot.plot(np.arange(len(bp_td_loss)), np.array(bp_td_loss), 'g-')
    plot.xlabel('iters')
    plot.ylabel('loss')
    plot.title('Before-passing touchdown Loss')
    plot.savefig('loss_vis/bp_td_loss.jpg')

    plot.figure(4)
    plot.plot(np.arange(len(ap_pred_loss)), np.array(ap_pred_loss), 'g-')
    plot.xlabel('iters')
    plot.ylabel('loss')
    plot.title('After-passing pred Loss')
    plot.savefig('loss_vis/ap_pred_loss.jpg')

    plot.figure(5)
    plot.plot(np.arange(len(bp_pred_loss)), np.array(bp_pred_loss), 'g-')
    plot.xlabel('iters')
    plot.ylabel('loss')
    plot.title('Before-passing pred Loss')
    plot.savefig('loss_vis/bp_pred_loss.jpg')

    # Testing predictor
    nums_seen = 0
    nums_x = 0
    acc_loss = 0
    td_correct = 0
    print('\nTesting before_passing predictor...')
    for i, (data, _, x_label, td_label) in enumerate(bp_teloader):
        # touchdown prediction
        touchdown = bp_td(data)
        touchdown = torch.argmax(touchdown, 1)

        # filter no-touchdown instances for x-predictor
        x_data = data[touchdown == 0]
        x_label = x_label[touchdown == 0]
        if len(x_data.shape) > 0:
            data_encoded, _ = bp_ae(x_data)
            logits = bp_pred(data_encoded)
            loss_val = pred_loss(logits, x_label)
            acc_loss += x_data.shape[0] * loss_val
            nums_x += x_data.shape[0]

        # collect metrics
        td_correct += torch.sum(touchdown == td_label.squeeze(1)).item()
        nums_seen += data.shape[0]

    print(f'BP testing results: x_pred_MSE={acc_loss / nums_x}\n',
          f'                   touchdown_precision={td_correct / nums_seen}')

    nums_seen = 0
    nums_x = 0
    acc_loss = 0
    td_correct = 0
    print('\nTesting after_passing predictor...')
    for i, (data, _, x_label, td_label) in enumerate(ap_teloader):
        # touchdown prediction
        touchdown = ap_td(data)
        touchdown = torch.argmax(touchdown, 1)

        # filter no-touchdown instances for x-predictor
        x_data = data[touchdown == 0]
        x_label = x_label[touchdown == 0]
        if len(x_data.shape) > 0:
            data_encoded, _ = ap_ae(x_data)
            logits = ap_pred(data_encoded)
            loss_val = pred_loss(logits, x_label)
            acc_loss += x_data.shape[0] * loss_val
            nums_x += x_data.shape[0]

        # collect metrics
        td_correct += torch.sum(touchdown == td_label.squeeze(1)).item()
        nums_seen += data.shape[0]

    print(f'AP testing results: x_pred_MSE={acc_loss / nums_x}\n',
          f'                   touchdown_precision={td_correct / nums_seen}')
