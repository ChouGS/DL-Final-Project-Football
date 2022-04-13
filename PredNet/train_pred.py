import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from pred_model import PredNet
from dataset import PredictorDataset

EPOCH = 10
BATCH_SIZE = 256
TEST_BSIZE = 1024
LR = 0.002
TEST_PORTION = 0.2
PRINT_FREQ = 50

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

ap_net = PredNet([128, 256, 256, 64, 8])
bp_net = PredNet([128, 256, 256, 64, 8])

loss = nn.MSELoss()

ap_opt = optim.Adam(ap_net.parameters(), lr=LR)
bp_opt = optim.Adam(bp_net.parameters(), lr=LR)

# Training
print('\nTraining before_passing network...')
niters = len(bp_train) // BATCH_SIZE + 1
for e in range(EPOCH):
    for i, (data, _, label) in enumerate(bp_trloader):
        logits = bp_net(data)
        
        loss_val = loss(logits, label)

        bp_opt.zero_grad()
        loss_val.backward()
        bp_opt.step()

        if i % PRINT_FREQ == 0:
            print(f'epoch {e}, iter {i + 1}/{niters}: bp_loss={loss_val}')

print('\nTraining after_passing network...')
niters = len(ap_train) // BATCH_SIZE + 1
for e in range(EPOCH):
    for i, (data, _, label) in enumerate(ap_trloader):
        logits = ap_net(data)
        
        loss_val = loss(logits, label)

        ap_opt.zero_grad()
        loss_val.backward()
        ap_opt.step()

        if i % PRINT_FREQ == 0:
            print(f'epoch {e}, iter {i + 1}/{niters}: ap_loss={loss_val}')

# Testing
bp_net.eval()
nums_seen = 0
acc_loss = 0
print('\nTesting before_passing network...')
for i, (data, _, label) in enumerate(bp_teloader):
    logits = bp_net(data)
    
    loss_val = loss(logits, label)

    nums_seen += data.shape[0]
    acc_loss += data.shape[0] * loss_val
print(f'Testing results: MSE={acc_loss / nums_seen}')

ap_net.eval()
nums_seen = 0
acc_loss = 0
print('\nTesting after_passing network...')
for i, (data, _, label) in enumerate(ap_teloader):
    logits = ap_net(data)
    
    loss_val = loss(logits, label)

    nums_seen += data.shape[0]
    acc_loss += data.shape[0] * loss_val
print(f'Testing results: MSE={acc_loss / nums_seen}')
