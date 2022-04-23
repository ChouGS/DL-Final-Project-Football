import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import os
import re
import shutil
import argparse
from pred_model import PredGAT

from tools import find_last_epoch
from pred_model import PredX, PredATT
from dataset import DecisionDataset
from vis import vis_loss_curve
from config.default import get_default_cfg

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load in-line hyper params
parser = argparse.ArgumentParser()
parser.add_argument("-cfg", "--config", help="[Required] the path to a .yaml file to use as the config.", \
                    type=str, required=True)
parser.add_argument("-cont", "--continue_training", help="[Optional] whether to resume from checkpoints.", \
                    required=False, action='store_true')
args = parser.parse_args()
args.config = re.sub('\\\\', '/', args.config)

# Load configuration hyper params
cfg = get_default_cfg()
cfg.merge_from_file(args.config)
cfg.freeze()

os.makedirs(f'PredNet/results/{cfg.NAME}/config', exist_ok=True)
os.makedirs(f'PredNet/results/{cfg.NAME}/models', exist_ok=True)
shutil.copy(args.config, f'PredNet/results/{cfg.NAME}/config/{args.config.split("/")[-1]}')

if __name__ == '__main__':
    # Load data
    ap_data = np.load(os.path.join(cfg.DATA_DIR, 'data_after_passing.npy'))

    off_index = [i + j for i in range(0, ap_data.shape[0], 22) for j in range(0, 11)]
    def_index = [i + j for i in range(0, ap_data.shape[0], 22) for j in range(11, 22)]
    off_data = ap_data[off_index]
    def_data = ap_data[def_index]

    off_train = DecisionDataset(off_data[:int(round(off_data.shape[0] * (1 - cfg.DATA.TESTRATIO)))])
    off_test = DecisionDataset(off_data[int(round(off_data.shape[0] * (1 - cfg.DATA.TESTRATIO))):])
    def_train = DecisionDataset(def_data[:int(round(def_data.shape[0] * (1 - cfg.DATA.TESTRATIO)))])
    def_test = DecisionDataset(def_data[int(round(def_data.shape[0] * (1 - cfg.DATA.TESTRATIO))):])

    off_trloader = DataLoader(off_train, batch_size=cfg.DATA.TRAINBS, shuffle=True)
    off_teloader = DataLoader(off_test, batch_size=cfg.DATA.TESTBS, shuffle=True)
    def_trloader = DataLoader(def_train, batch_size=cfg.DATA.TRAINBS, shuffle=True)
    def_teloader = DataLoader(def_test, batch_size=cfg.DATA.TESTBS, shuffle=True)

    # Load pretrained predictor network
    if cfg.PRED == 'MLP':
        ap_pred = PredX(cfg.MODEL.X)
    elif cfg.PRED == 'ATT':
        ap_pred = PredATT(cfg.MODEL.ATT)
    ap_pred.load_state_dict(cfg.PRETRAIN_DIR)
    ap_pred.eval()

    ### Train offensive GAT model
    # Define GAT model
    off_gat = PredGAT(cfg.MODEL.GAT)

    # Find continuing epoch
    start_epoch = 0
    if args.continue_training:
        start_epoch = find_last_epoch(f'DeciNet/results/{cfg.NAME}/models/off_gat')
        if os.path.exists(f'DeciNet/results/{cfg.NAME}/models/off_gat/off_gat_{start_epoch}.th'):
            off_gat.load_state_dict(f'DeciNet/results/{cfg.NAME}/models/def_gat/off_gat_{start_epoch}.th')    

    # Begin training
    off_score_loss = []
    off_direction_loss = []
    off_velo_loss = []
    off_gat_loss = []
    epoch = cfg.MODEL.GAT.EPOCH
    print('\nTraining after_passing predictor...')
    niters = len(off_train) // cfg.DATA.TRAINBS + 1
    for e in range(start_epoch, epoch):
        for i, (data, pos, v, x_score) in enumerate(off_trloader):
            # GAT forward
            decision = off_gat(data)
            
            # score loss
            pred_data = torch.concat((data, pos, decision), 2)
            score_loss_val = ap_pred(pred_data)
            off_score_loss.append(score_loss_val.item())

            ap_pred_opt.zero_grad()
            loss_val.backward()
            ap_pred_opt.step()

            if i % cfg.PRINT_FREQ == 0:
                print(f'Pred epoch {e}/{epoch}, iter {i + 1}/{niters}: ap_pred_loss={loss_val}')
                
        if e % cfg.SAVE_FREQ == 0:
            torch.save(off_gat.state_dict(), f'PredNet/results/{cfg.NAME}/models/pred/ap_pred_{e}.th')
    torch.save(off_gat.state_dict(), f'PredNet/results/{cfg.NAME}/models/pred/ap_pred_final.th')

    loss_dict = {
        'off_score_loss': off_score_loss,
        'off_direction_loss': off_direction_loss,
        'off_velo_loss': off_velo_loss,
        'off_gat_loss': off_gat_loss,
    }

    # Plot loss curves
    vis_loss_curve(cfg, loss_dict)

    # Testing predictor
    nums_seen = 0
    nums_x = 0
    acc_loss = 0
    td_correct = 0
    print('\nTesting before_passing predictor...')
    for i, (data, _, x_score, td_label) in enumerate(bp_teloader):
        
        logits = bp_pred(data)
        loss_val = pred_loss(logits, x_score)
        acc_loss += data.shape[0] * loss_val
        nums_x += data.shape[0]

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
    for i, (data, _, x_score, td_label) in enumerate(off_teloader):
        
        logits = ap_pred(data)
        loss_val = pred_loss(logits, x_score)
        acc_loss += data.shape[0] * loss_val
        nums_x += data.shape[0]

        # collect metrics
        nums_seen += data.shape[0]

    print(f'AP testing results: x_pred_MSE={acc_loss / nums_x}\n')
    if cfg.MTASK:
        print(f'                    touchdown_precision={td_correct / nums_seen}')
