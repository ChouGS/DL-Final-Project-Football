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
from tools import find_last_epoch, make_pred_data, make_gat_data
from loss import DirectionLoss, VeloLoss
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
                    type=bool, required=False, default=False)
args = parser.parse_args()
args.config = re.sub('\\\\', '/', args.config)

# Load configuration hyper params
cfg = get_default_cfg()
cfg.merge_from_file(args.config)
cfg.freeze()

os.makedirs(f'DeciNet/results/{cfg.NAME}/config', exist_ok=True)
os.makedirs(f'DeciNet/results/{cfg.NAME}/models', exist_ok=True)
shutil.copy(args.config, f'DeciNet/results/{cfg.NAME}/config/{args.config.split("/")[-1]}')

if __name__ == '__main__':
    # Load data
    ap_data = np.load(os.path.join(cfg.DATA_DIR, 'data_after_passing.npy'))

    off_index = [i + j for i in range(0, ap_data.shape[0], 22) for j in range(0, 11)]
    def_index = [i + j for i in range(0, ap_data.shape[0], 22) for j in range(11, 22)]
    off_data = ap_data[off_index]
    def_data = ap_data[def_index]

    # off_groupid = np.array([i // 11 for i in range(off_data.shape[0])], dtype=np.int32)[:, np.newaxis]
    # off_data = np.concatenate([off_data, off_groupid], 1)
    # prev_v_all = off_data[:, 71:73]
    # prev_v_off = np.reshape(prev_v_all, (-1, 11, 2))

    # def_groupid = np.array([i // 11 for i in range(def_data.shape[0])], dtype=np.int32)[:, np.newaxis]
    # def_data = np.concatenate([def_data, def_groupid], 1)
    # prev_v_all = def_data[:, 71:73]
    # prev_v_def = np.reshape(prev_v_all, (-1, 11, 2))

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
    ap_pred.load_state_dict(torch.load(open(cfg.PRETRAIN_DIR, 'rb')))
    ap_pred.eval()

    ### Train offensive GAT model
    # Define GAT model
    off_gat = PredGAT(cfg.MODEL.GAT)

    # Create save_dir
    os.makedirs(f'DeciNet/results/{cfg.NAME}/models/off_gat', exist_ok=True)

    # Find continuing epoch
    start_epoch = 0
    if args.continue_training:
        start_epoch, sd_path = find_last_epoch(f'DeciNet/results/{cfg.NAME}/models/off_gat')
        if start_epoch == -1:
            start_epoch = cfg.MODEL.GAT.EPOCH
            off_gat = torch.load(sd_path)
        elif sd_path is not None:
            state_dict = torch.load(sd_path)
            off_gat.load_state_dict(state_dict)   

    # Optimizer
    off_optim = optim.Adam(off_gat.parameters(), cfg.MODEL.GAT.LR)

    # Loss functions
    dir_loss = DirectionLoss()
    velo_loss = VeloLoss()
    dir_loss.eval()
    velo_loss.eval()

    # Begin training
    off_score_loss = []
    off_direction_loss = []
    off_velo_loss = []
    off_gat_loss = []
    epoch = cfg.MODEL.GAT.EPOCH
    print('\nTraining offensive decisionmaker...')
    niters = len(off_train) // cfg.DATA.TRAINBS + 1
    for e in range(start_epoch, epoch):
        for i, (data, pos, v) in enumerate(off_trloader):
            # GAT forward
            gat_data = make_gat_data(data, pos, v)
            off_decision = off_gat(gat_data)
            off_data = make_pred_data(data, pos, off_decision)

            # score loss
            off_score_loss_val = torch.mean(ap_pred(off_data))
            off_score_loss.append(off_score_loss_val.item())

            off_velo_loss_val = velo_loss(off_decision)
            off_velo_loss.append(off_velo_loss_val.item())

            off_direction_loss_val = dir_loss(off_decision, v)
            off_direction_loss.append(off_direction_loss_val.item())

            off_gat_loss_val = -cfg.W_SCORE * off_score_loss_val + cfg.W_DIRE * off_direction_loss_val + cfg.W_VELO * off_velo_loss_val
            off_gat_loss.append(off_gat_loss_val.item())

            off_optim.zero_grad()
            off_gat_loss_val.backward()
            off_optim.step()

            if i % cfg.PRINT_FREQ == 0:
                print(f'Offensive epoch {e}/{epoch}, iter {i + 1}/{niters}: \n', 
                      f'   total loss={round(off_gat_loss_val.item(), 3)}',
                      f'   score loss={round(off_score_loss_val.item(), 3)}',
                      f'   direction loss={round(off_direction_loss_val.item(), 3)}',
                      f'   velo loss={round(off_velo_loss_val.item(), 3)}\n')
                
        if e % cfg.SAVE_FREQ == 0:
            torch.save(off_gat.state_dict(), f'DeciNet/results/{cfg.NAME}/models/off_gat/off_gat_{e}.th')
    torch.save(off_gat.state_dict(), f'DeciNet/results/{cfg.NAME}/models/off_gat/off_gat_final.th')
    
    
    
    ### Train defensive GAT model
    # Define GAT model
    def_gat = PredGAT(cfg.MODEL.GAT)

    # Create save_dir
    os.makedirs(f'DeciNet/results/{cfg.NAME}/models/def_gat', exist_ok=True)

    # Find continuing epoch
    start_epoch = 0
    if args.continue_training:
        start_epoch, sd_path = find_last_epoch(f'DeciNet/results/{cfg.NAME}/models/def_gat')
        if start_epoch == -1:
            start_epoch = cfg.MODEL.GAT.EPOCH
            def_gat = torch.load(sd_path)
        elif sd_path is not None:
            state_dict = torch.load(sd_path)
            def_gat.load_state_dict(state_dict)    

    # Optimizer
    def_optim = optim.Adam(def_gat.parameters(), cfg.MODEL.GAT.LR)

    # Loss functions
    dir_loss = DirectionLoss()
    velo_loss = VeloLoss()

    # Begin training
    def_score_loss = []
    def_direction_loss = []
    def_velo_loss = []
    def_gat_loss = []
    epoch = cfg.MODEL.GAT.EPOCH
    print('\nTraining defensive decisionmaker...')
    niters = len(def_train) // cfg.DATA.TRAINBS + 1
    for e in range(start_epoch, epoch):
        for i, (data, pos, v) in enumerate(def_trloader):
            # GAT forward
            gat_data = make_gat_data(data, pos, v)
            def_decision = def_gat(gat_data)
            def_data = make_pred_data(data, pos, def_decision)

            # score loss
            def_score_loss_val = torch.mean(ap_pred(def_data))
            def_score_loss.append(def_score_loss_val.item())

            def_velo_loss_val = velo_loss(def_decision)
            def_velo_loss.append(def_velo_loss_val.item())

            def_direction_loss_val = dir_loss(def_decision, v)
            def_direction_loss.append(def_direction_loss_val.item())

            def_gat_loss_val = cfg.W_SCORE * def_score_loss_val + cfg.W_DIRE * def_direction_loss_val + cfg.W_VELO * def_velo_loss_val
            def_gat_loss.append(def_gat_loss_val.item())

            def_optim.zero_grad()
            def_gat_loss_val.backward()
            def_optim.step()

            if i % cfg.PRINT_FREQ == 0:
                print(f'Defensive epoch {e}/{epoch}, iter {i + 1}/{niters}: \n', 
                      f'   total loss={round(def_gat_loss_val.item(), 3)}',
                      f'   score loss={round(def_score_loss_val.item(), 3)}',
                      f'   direction loss={round(def_direction_loss_val.item(), 3)}',
                      f'   velo loss={round(def_velo_loss_val.item(), 3)}\n')
                
        if e % cfg.SAVE_FREQ == 0:
            torch.save(def_gat.state_dict(), f'DeciNet/results/{cfg.NAME}/models/def_gat/def_gat_{e}.th')
    torch.save(def_gat.state_dict(), f'DeciNet/results/{cfg.NAME}/models/def_gat/def_gat_final.th')

    loss_dict = {
        'off_score_loss': off_score_loss,
        'off_velo_loss': off_velo_loss,
        'off_direction_loss': off_direction_loss,
        'off_gat_loss': off_gat_loss,
        'def_score_loss': def_score_loss,
        'def_direction_loss': def_direction_loss,
        'def_velo_loss': def_velo_loss,
        'def_gat_loss': def_gat_loss
    }

    # Plot loss curves
    vis_loss_curve(cfg, loss_dict)

    # Freeze models
    off_gat.eval()
    def_gat.eval()

    # Testing offensive decisionmaker
    nums_seen = 0
    acc_gat_loss = 0
    acc_score_loss = 0
    acc_direction_loss = 0
    acc_velo_loss = 0
    print('\nTesting offensive decisionmaker...')
    for i, (data, pos, v) in enumerate(off_teloader):
        # GAT forward
        gat_data = make_gat_data(data, pos, v)
        off_decision = def_gat(gat_data)
        off_data = make_pred_data(data, pos, off_decision)

        # score loss
        off_score_loss_val = torch.sum(ap_pred(off_data))
        acc_score_loss += off_score_loss_val.item()

        off_velo_loss_val = velo_loss(off_decision)
        acc_velo_loss += off_velo_loss_val.item()

        off_direction_loss_val = dir_loss(off_decision, v)
        acc_direction_loss += off_direction_loss_val.item()

        off_gat_loss_val = -cfg.W_SCORE * off_score_loss_val + cfg.W_DIRE * off_direction_loss_val + cfg.W_VELO * off_velo_loss_val
        acc_gat_loss += off_gat_loss_val.item()

        nums_seen += off_data.shape[0]

    print(f'Offensive decision testing results: \n',
          f'   total loss={round(acc_gat_loss / nums_seen, 3)}',
          f'   score loss={round(acc_score_loss / nums_seen, 3)}',
          f'   direction loss={round(acc_direction_loss / nums_seen, 3)}',
          f'   velo loss={round(acc_velo_loss / nums_seen, 3)}\n')

    # Testing defensive decisionmaker
    nums_seen = 0
    acc_gat_loss = 0
    acc_score_loss = 0
    acc_velo_loss = 0
    print('\nTesting defensive decisionmaker...')
    for i, (data, pos, v) in enumerate(def_teloader):
        # GAT forward
        gat_data = make_gat_data(data, pos, v)
        def_decision = def_gat(gat_data)
        def_data = make_pred_data(data, pos, def_decision)

        # score loss
        def_score_loss_val = torch.sum(ap_pred(def_data))
        acc_score_loss += def_score_loss_val.item()

        def_velo_loss_val = velo_loss(def_decision)
        acc_velo_loss += def_velo_loss_val.item()
        
        def_direction_loss_val = dir_loss(def_decision, v)
        acc_direction_loss += def_direction_loss_val.item()

        def_gat_loss_val = cfg.W_SCORE * def_score_loss_val + cfg.W_DIRE * def_direction_loss_val + cfg.W_VELO * def_velo_loss_val
        acc_gat_loss += def_gat_loss_val.item()

        nums_seen += def_data.shape[0]

    print(f'Defensive decision testing results: \n',
          f'   total loss={round(acc_gat_loss / nums_seen, 3)}',
          f'   score loss={round(acc_score_loss / nums_seen, 3)}',
          f'   direction loss={round(acc_direction_loss / nums_seen, 3)}',
          f'   velo loss={round(acc_velo_loss / nums_seen, 3)}\n')
