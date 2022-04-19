import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from pred_model import PredAE, PredNet
from dataset import PredictorDataset

EPOCH = 10
BATCH_SIZE = 256
TEST_BSIZE = 1024
LR = 0.002
TEST_PORTION = 0.2
PRINT_FREQ = 50

if __name__ == '__main__':
    # Load data
    ap_data = np.load('data_after_passing.npy')
    bp_data = np.load('data_before_passing.npy')

    ap_train = PredictorDataset(ap_data[:int(round(ap_data.shape[0] * (1 - TEST_PORTION)))])
    bp_train = PredictorDataset(bp_data[:int(round(bp_data.shape[0] * (1 - TEST_PORTION)))])
    ap_test = PredictorDataset(ap_data[int(round(ap_data.shape[0] * (1 - TEST_PORTION))):])
    bp_test = PredictorDataset(bp_data[int(round(bp_data.shape[0] * (1 - TEST_PORTION))):])

    ap_trloader = DataLoader(ap_train, batch_size=BATCH_SIZE)
    bp_trloader = DataLoader(bp_train, batch_size=BATCH_SIZE)
    ap_teloader = DataLoader(ap_test, batch_size=TEST_BSIZE)
    bp_teloader = DataLoader(bp_test, batch_size=TEST_BSIZE)

    # Define autoencoder model
    ap_ae = PredAE([128, 256, 256, 64])
    bp_ae = PredAE([128, 256, 256, 64])

    # Reconstruction loss
    rec_loss = nn.MSELoss()

    # Optimizer
    ap_ae_opt = optim.Adam(ap_ae.parameters(), lr=LR)
    bp_ae_opt = optim.Adam(bp_ae.parameters(), lr=LR)

    # Training autoencoder
    print('\nTraining before_passing autoencoder...')
    niters = len(bp_train) // BATCH_SIZE + 1
    for e in range(EPOCH):
        for i, (data, _, label) in enumerate(bp_trloader):
            _, data_rec = bp_ae(data)
            
            loss_val = rec_loss(data_rec, data)

            bp_ae_opt.zero_grad()
            loss_val.backward()
            bp_ae_opt.step()

            if i % PRINT_FREQ == 0:
                print(f'AE epoch {e}/{EPOCH}, iter {i + 1}/{niters}: bp_rec_loss={loss_val}')

    print('\nTraining after_passing autoencoder...')
    niters = len(ap_train) // BATCH_SIZE + 1
    for e in range(EPOCH):
        for i, (data, _, label) in enumerate(ap_trloader):
            _, data_rec = ap_ae(data)
            
            loss_val = rec_loss(data_rec, data)

            ap_ae_opt.zero_grad()
            loss_val.backward()
            ap_ae_opt.step()

            if i % PRINT_FREQ == 0:
                print(f'AE epoch {e}/{EPOCH}, iter {i + 1}/{niters}: ap_rec_loss={loss_val}')

    # Using the trained autoencoder to train a predictor
    pred_loss = nn.MSELoss()

    bp_ae.eval()
    bp_pred = PredNet([64, 256, 256, 64, 16], 16)
    bp_pred_opt = optim.Adam(bp_ae.parameters(), lr=LR)
    
    print('\nTraining before_passing predictor...')
    niters = len(bp_train) // BATCH_SIZE + 1
    for e in range(EPOCH):
        for i, (data, _, label) in enumerate(bp_trloader):
            data_encoded, _ = bp_ae(data)
            logits = bp_pred(data_encoded)
            
            loss_val = pred_loss(logits, label)

            bp_pred_opt.zero_grad()
            loss_val.backward()
            bp_pred_opt.step()

            if i % PRINT_FREQ == 0:
                print(f'Pred epoch {e}/{EPOCH}, iter {i + 1}/{niters}: bp_rec_loss={loss_val}')

    ap_ae.eval()
    ap_pred = PredNet([64, 256, 256, 64, 16], 16)
    ap_pred_opt = optim.Adam(ap_ae.parameters(), lr=LR)

    print('\nTraining after_passing predictor...')
    niters = len(ap_train) // BATCH_SIZE + 1
    for e in range(EPOCH):
        for i, (data, _, label) in enumerate(ap_trloader):
            data_encoded, _ = ap_ae(data)
            logits = ap_pred(data_encoded)
            
            loss_val = pred_loss(logits, label)

            ap_pred_opt.zero_grad()
            loss_val.backward()
            ap_pred_opt.step()

            if i % PRINT_FREQ == 0:
                print(f'Pred epoch {e}/{EPOCH}, iter {i + 1}/{niters}: ap_rec_loss={loss_val}')
    
    # Testing predictor
    nums_seen = 0
    acc_loss = 0
    print('\nTesting before_passing predictor...')
    for i, (data, _, label) in enumerate(bp_teloader):
        logits = bp_ae(data)
        
        loss_val = pred_loss(logits, label)

        nums_seen += data.shape[0]
        acc_loss += data.shape[0] * loss_val
    print(f'Testing results: MSE={acc_loss / nums_seen}')

    ap_ae.eval()
    nums_seen = 0
    acc_loss = 0
    print('\nTesting after_passing predictor...')
    for i, (data, _, label) in enumerate(ap_teloader):
        logits = ap_ae(data)
        
        loss_val = pred_loss(logits, label)

        nums_seen += data.shape[0]
        acc_loss += data.shape[0] * loss_val
    print(f'Testing results: MSE={acc_loss / nums_seen}')
