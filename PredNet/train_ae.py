import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import os
import re
import shutil
import argparse

from pred_model import PredAE, PredX, PredTD, PredATT
from dataset import PredictorDataset, AttentionDataset
from vis import vis_loss_curve
from config.default import get_default_cfg

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser()
parser.add_argument("-cfg", "--config", help="[Required] the path to a .yaml file to use as the config.", \
                    type=str, required=True)
args = parser.parse_args()
args.config = re.sub('\\\\', '/', args.config)

cfg = get_default_cfg()
cfg.merge_from_file(args.config)
os.makedirs(f'PredNet/results/{cfg.NAME}/config', exist_ok=True)
os.makedirs(f'PredNet/results/{cfg.NAME}/models', exist_ok=True)
shutil.copy(args.config, f'PredNet/results/{cfg.NAME}/config/{args.config.split("/")[-1]}')

if __name__ == '__main__':
    # Load data
    ap_data = np.load(os.path.join(cfg.DATA_DIR, 'data_after_passing.npy'))
    bp_data = np.load(os.path.join(cfg.DATA_DIR, 'data_before_passing.npy'))

    if cfg.PRED == 'MLP':
        ap_train = PredictorDataset(ap_data[:int(round(ap_data.shape[0] * (1 - cfg.DATA.TESTRATIO)))])
        bp_train = PredictorDataset(bp_data[:int(round(bp_data.shape[0] * (1 - cfg.DATA.TESTRATIO)))])
        ap_test = PredictorDataset(ap_data[int(round(ap_data.shape[0] * (1 - cfg.DATA.TESTRATIO))):])
        bp_test = PredictorDataset(bp_data[int(round(bp_data.shape[0] * (1 - cfg.DATA.TESTRATIO))):])
    elif cfg.PRED == 'ATT':
        ap_train = AttentionDataset(ap_data[:int(round(ap_data.shape[0] * (1 - cfg.DATA.TESTRATIO)))])
        bp_train = AttentionDataset(bp_data[:int(round(bp_data.shape[0] * (1 - cfg.DATA.TESTRATIO)))])
        ap_test = AttentionDataset(ap_data[int(round(ap_data.shape[0] * (1 - cfg.DATA.TESTRATIO))):])
        bp_test = AttentionDataset(bp_data[int(round(bp_data.shape[0] * (1 - cfg.DATA.TESTRATIO))):])
    else:
        raise NotImplementedError(f'Not supported predictor type {cfg.PRED}')

    ap_trloader = DataLoader(ap_train, batch_size=cfg.DATA.TRAINBS, shuffle=True)
    bp_trloader = DataLoader(bp_train, batch_size=cfg.DATA.TRAINBS, shuffle=True)
    ap_teloader = DataLoader(ap_test, batch_size=cfg.DATA.TESTBS, shuffle=True)
    bp_teloader = DataLoader(bp_test, batch_size=cfg.DATA.TESTBS, shuffle=True)

    # Load data for multitasking
    if cfg.MTASK:
        ap_tr_notd = PredictorDataset(ap_data[ap_data[:, -1] == 0])
        bp_tr_notd = PredictorDataset(bp_data[bp_data[:, -1] == 0])
        ap_tr_notd_loader = DataLoader(ap_tr_notd, batch_size=cfg.DATA.TRAINBS, shuffle=True)    # No-touchdown data
        bp_tr_notd_loader = DataLoader(bp_tr_notd, batch_size=cfg.DATA.TRAINBS, shuffle=True)    # No-touchdown data
        cfg.MODEL.TD.IN_DIM = cfg.DATA.FDIM
    else:
        ap_tr_notd = ap_train
        bp_tr_notd = bp_train
        ap_tr_notd_loader = ap_trloader
        bp_tr_notd_loader = bp_trloader

    cfg.MODEL.X.IN_DIM = cfg.MODEL.AE.LAT_D if cfg.USE_AE else cfg.DATA.FDIM
    if cfg.USE_AE:
        cfg.MODEL.AE.IN_DIM = cfg.DATA.FDIM

    cfg.freeze()

    ### Training autoencoder network
    if cfg.USE_AE:
        # Define autoencoder model
        ap_ae = PredAE(cfg.MODEL.AE)
        bp_ae = PredAE(cfg.MODEL.AE)

        # Reconstruction loss
        rec_loss = nn.MSELoss()

        # Optimizer
        ap_ae_opt = optim.Adam(ap_ae.parameters(), lr=cfg.MODEL.AE.LR)
        bp_ae_opt = optim.Adam(bp_ae.parameters(), lr=cfg.MODEL.AE.LR)

        # Training
        ap_ae_loss = []
        bp_ae_loss = []
        print('\nTraining before_passing autoencoder...')
        niters = len(bp_train) // cfg.DATA.TRAINBS + 1
        for e in range(cfg.MODEL.AE.EPOCH):
            for i, (data, _, _, _) in enumerate(bp_trloader):
                _, data_rec = bp_ae(data)
                
                loss_val = rec_loss(data_rec, data)
                bp_ae_loss.append(loss_val.item())

                bp_ae_opt.zero_grad()
                loss_val.backward()
                bp_ae_opt.step()

                if i % cfg.PRINT_FREQ == 0:
                    print(f'AE epoch {e}/{cfg.MODEL.AE.EPOCH}, iter {i + 1}/{niters}: bp_rec_loss={loss_val}')

        print('\nTraining after_passing autoencoder...')
        niters = len(ap_train) // cfg.DATA.TRAINBS + 1
        for e in range(cfg.MODEL.AE.EPOCH):
            for i, (data, _, _, _) in enumerate(ap_trloader):
                _, data_rec = ap_ae(data)
                
                loss_val = rec_loss(data_rec, data)
                ap_ae_loss.append(loss_val.item())

                ap_ae_opt.zero_grad()
                loss_val.backward()
                ap_ae_opt.step()

                if i % cfg.PRINT_FREQ == 0:
                    print(f'AE epoch {e}/{cfg.MODEL.AE.EPOCH}, iter {i + 1}/{niters}: ap_rec_loss={loss_val}')

    ### Train touchdown network
    if cfg.MTASK:
        # Define touchdown predictor
        ap_td = PredTD(cfg.MODEL.TD)
        bp_td = PredTD(cfg.MODEL.TD)

        # Optimizer
        ap_td_opt = optim.Adam(ap_td.parameters(), lr=cfg.MODEL.TD.LR)
        bp_td_opt = optim.Adam(bp_td.parameters(), lr=cfg.MODEL.TD.LR)

        # Loss
        td_bce_loss = nn.BCELoss()

        # Training
        ap_td_loss = []
        bp_td_loss = []
        print('\nTraining before_passing touchdown...')
        niters = len(bp_train) // cfg.DATA.TRAINBS + 1
        for e in range(cfg.MODEL.TD.EPOCH):
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

                if i % cfg.PRINT_FREQ == 0:
                    print(f'TD epoch {e}/{cfg.MODEL.TD.EPOCH}, iter {i + 1}/{niters}: bp_td_loss={loss_val}')

        print('\nTraining after_passing touchdown...')
        niters = len(ap_train) // cfg.DATA.TRAINBS + 1
        for e in range(cfg.MODEL.TD.EPOCH):
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

                if i % cfg.PRINT_FREQ == 0:
                    print(f'TD epoch {e}/{cfg.MODEL.TD.EPOCH}, iter {i + 1}/{niters}: ap_td_loss={loss_val}')

        bp_ae.eval()
        ap_ae.eval()

    # Using the trained autoencoder to train a predictor (only on no-touchdown model)
    pred_loss = nn.MSELoss()
    if cfg.PRED == 'MLP':
        bp_pred = PredX(cfg.MODEL.X)
        ap_pred = PredX(cfg.MODEL.X)
        ap_pred_opt = optim.Adam(ap_pred.parameters(), lr=cfg.MODEL.X.LR)
        bp_pred_opt = optim.Adam(bp_pred.parameters(), lr=cfg.MODEL.X.LR)
        epoch = cfg.MODEL.X.EPOCH
    elif cfg.PRED == 'ATT':
        bp_pred = PredATT(cfg.MODEL.ATT)
        ap_pred = PredATT(cfg.MODEL.ATT)
        ap_pred_opt = optim.Adam(ap_pred.parameters(), lr=cfg.MODEL.ATT.LR)
        bp_pred_opt = optim.Adam(bp_pred.parameters(), lr=cfg.MODEL.ATT.LR)
        epoch = cfg.MODEL.ATT.EPOCH
    
    ap_pred_loss = []
    bp_pred_loss = []
    print('\nTraining before_passing predictor...')
    niters = len(bp_tr_notd) // cfg.DATA.TRAINBS + 1
    for e in range(0):
        for i, (data, _, x_label, _) in enumerate(bp_tr_notd_loader):
            if cfg.USE_AE:
                data_encoded, _ = bp_ae(data)
            else:
                data_encoded = data

            logits = bp_pred(data_encoded)
            
            loss_val = pred_loss(logits, x_label)
            bp_pred_loss.append(loss_val.item())

            bp_pred_opt.zero_grad()
            loss_val.backward()
            bp_pred_opt.step()

            if i % cfg.PRINT_FREQ == 0:
                print(f'Pred epoch {e}/{epoch}, iter {i + 1}/{niters}: bp_pred_loss={loss_val}')

    print('\nTraining after_passing predictor...')
    niters = len(ap_tr_notd) // cfg.DATA.TRAINBS + 1
    for e in range(epoch):
        for i, (data, _, x_label, _) in enumerate(ap_tr_notd_loader):
            if cfg.USE_AE:
                data_encoded, _ = ap_ae(data)
            else:
                data_encoded = data

            logits = ap_pred(data_encoded)
            
            loss_val = pred_loss(logits, x_label)
            ap_pred_loss.append(loss_val.item())

            ap_pred_opt.zero_grad()
            loss_val.backward()
            ap_pred_opt.step()

            if i % cfg.PRINT_FREQ == 0:
                print(f'Pred epoch {e}/{epoch}, iter {i + 1}/{niters}: ap_pred_loss={loss_val}')
    
    loss_dict = {
        'ap_x_loss': ap_pred_loss,
        'bp_x_loss': bp_pred_loss
    }
    if cfg.USE_AE:
        loss_dict['ap_ae_loss'] = ap_ae_loss
        loss_dict['bp_ae_loss'] = bp_ae_loss
    if cfg.MTASK:
        loss_dict['ap_td_loss'] = ap_td_loss
        loss_dict['bp_td_loss'] = bp_td_loss

    # Plot loss curves
    vis_loss_curve(cfg, loss_dict)

    # Testing predictor
    nums_seen = 0
    nums_x = 0
    acc_loss = 0
    td_correct = 0
    print('\nTesting before_passing predictor...')
    for i, (data, _, x_label, td_label) in enumerate(bp_teloader):
        if cfg.MTASK:
            # touchdown prediction
            touchdown = bp_td(data)
            touchdown = torch.argmax(touchdown, 1)

            # filter no-touchdown instances for x-predictor
            x_data = data[touchdown == 0]
            x_label = x_label[touchdown == 0]

            td_correct += torch.sum(touchdown == td_label.squeeze(1)).item()
        else:
            x_data = data

        if len(x_data.shape) > 0:
            if cfg.USE_AE:
                data_encoded, _ = bp_ae(x_data)
            else:
                data_encoded = data
            logits = bp_pred(data_encoded)
            loss_val = pred_loss(logits, x_label)
            acc_loss += x_data.shape[0] * loss_val
            nums_x += x_data.shape[0]

        # collect metrics
        nums_seen += data.shape[0]

    print(f'BP testing results: x_pred_MSE={acc_loss / nums_x}\n')
    if cfg.MTASK:
        print(f'                    touchdown_precision={td_correct / nums_seen}')

    nums_seen = 0
    nums_x = 0
    acc_loss = 0
    td_correct = 0
    print('\nTesting after_passing predictor...')
    for i, (data, _, x_label, td_label) in enumerate(ap_teloader):
        if cfg.MTASK:
            # touchdown prediction
            touchdown = ap_td(data)
            touchdown = torch.argmax(touchdown, 1)

            # filter no-touchdown instances for x-predictor
            x_data = data[touchdown == 0]
            x_label = x_label[touchdown == 0]

            td_correct += torch.sum(touchdown == td_label.squeeze(1)).item()
        else:
            x_data = data

        if len(x_data.shape) > 0:
            if cfg.USE_AE:
                data_encoded, _ = ap_ae(x_data)
            else:
                data_encoded = data
            logits = ap_pred(data_encoded)
            loss_val = pred_loss(logits, x_label)
            acc_loss += x_data.shape[0] * loss_val
            nums_x += x_data.shape[0]

        # collect metrics
        nums_seen += data.shape[0]

    print(f'AP testing results: x_pred_MSE={acc_loss / nums_x}\n')
    if cfg.MTASK:
        print(f'                    touchdown_precision={td_correct / nums_seen}')

    os.makedirs(f'PredNet/results/{cfg.NAME}/models', exist_ok=True)
    if cfg.MTASK:
        torch.save(ap_td.state_dict(), f'PredNet/results/{cfg.NAME}/models/ap_td.th')
        torch.save(bp_td.state_dict(), f'PredNet/results/{cfg.NAME}/models/bp_td.th')
    if cfg.USE_AE:
        torch.save(ap_ae.state_dict(), f'PredNet/results/{cfg.NAME}/models/ap_ae.th')
        torch.save(bp_ae.state_dict(), f'PredNet/results/{cfg.NAME}/models/bp_ae.th')
    torch.save(ap_pred.state_dict(), f'PredNet/results/{cfg.NAME}/models/ap_pred.th')
    torch.save(bp_pred.state_dict(), f'PredNet/results/{cfg.NAME}/models/bp_pred.th')
    