import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import os
import re
import shutil
import argparse

from pred_model import GameGAT
from tools import find_last_epoch, make_pred_data, make_gat_data
from loss import DirectionLoss, VeloLoss, OOBLoss
from pred_model import PredX, ScoreATT
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
    qb_data = np.load(os.path.join(cfg.DATA_DIR, 'data_after_passing_qb.npy'))
    wr_data = np.load(os.path.join(cfg.DATA_DIR, 'data_after_passing_wr.npy'))
    to_data = np.load(os.path.join(cfg.DATA_DIR, 'data_after_passing_to.npy'))
    cb_data = np.load(os.path.join(cfg.DATA_DIR, 'data_after_passing_cb.npy'))
    td_data = np.load(os.path.join(cfg.DATA_DIR, 'data_after_passing_td.npy'))
    sf_data = np.load(os.path.join(cfg.DATA_DIR, 'data_after_passing_sf.npy'))

    best_speed_dict = {
        'cb': 10 * 0.75,
        'wr': 10 * 0.75,
        'sf': 10 * 0.75,
        'td': 4.5 * 0.75,
        'to': 4.5 * 0.75,
        'qb': 6 * 0.75
    }

    # Load pretrained predictor network
    if cfg.SCORENET == 'MLP':
        ap_pred = PredX(cfg.MODEL.X)
    elif cfg.SCORENET == 'ATT':
        ap_pred = ScoreATT(cfg.MODEL.ATT)
    ap_pred.load_state_dict(torch.load(open(cfg.PRETRAIN_DIR, 'rb')))
    ap_pred.eval()

    ### Train GAT model for each separate position
    for name in ['qb', 'wr', 'to', 'cb', 'td', 'sf']:
        # Create save_dir
        os.makedirs(f'DeciNet/results/{cfg.NAME}/models/{name}', exist_ok=True)

        train_dataset = DecisionDataset(eval(f'{name}_data')[:int(round(eval(f'{name}_data').shape[0] * (1 - cfg.DATA.TESTRATIO)))])
        test_dataset = DecisionDataset(eval(f'{name}_data')[int(round(eval(f'{name}_data').shape[0] * (1 - cfg.DATA.TESTRATIO))):])

        train_loader = DataLoader(train_dataset, batch_size=cfg.DATA.TRAINBS, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=cfg.DATA.TESTBS, shuffle=True)

        # Define GAT model
        gat = GameGAT(cfg.MODEL.GAT)

        # Find continuing epoch
        start_epoch = 0
        if args.continue_training:
            start_epoch, sd_path = find_last_epoch(f'DeciNet/results/{cfg.NAME}/models/{name}')
            if start_epoch == -1:
                start_epoch = cfg.MODEL.GAT.EPOCH
                state_dict = torch.load(sd_path)
                gat.load_state_dict(state_dict)
            elif sd_path is not None:
                state_dict = torch.load(sd_path)
                gat.load_state_dict(state_dict)   

        # Optimizer
        optimizer = optim.Adam(gat.parameters(), cfg.MODEL.GAT.LR)

        # Loss functions
        dir_loss_f = DirectionLoss()
        oob_loss_f = OOBLoss()
        velo_loss_f = VeloLoss(best_v=best_speed_dict[name])
        dir_loss_f.eval()
        velo_loss_f.eval()
        oob_loss_f.eval()

        # Begin training
        score_loss = []
        direction_loss = []
        velo_loss = []
        oob_loss = []
        gat_loss = []
        epoch = cfg.MODEL.GAT.EPOCH
        print(f'\nTraining {name} decisionmaker...')
        niters = len(train_dataset) // cfg.DATA.TRAINBS + 1
        for e in range(start_epoch, epoch):
            for i, (data, pos, v) in enumerate(train_loader):
                # GAT forward
                gat_data = make_gat_data(data, pos, v)
                decision = gat(gat_data)

                data = make_pred_data(data, pos, decision)

                # score loss
                score_loss_val = torch.mean(ap_pred(data))
                score_loss.append(score_loss_val.item())

                # velocity loss
                velo_loss_val = velo_loss_f(decision)
                velo_loss.append(velo_loss_val.item())

                # velocity loss
                dir_loss_val = dir_loss_f(decision, v)
                direction_loss.append(velo_loss_val.item())

                # out of bound loss
                oob_loss_val = oob_loss_f(pos, decision)
                oob_loss.append(oob_loss_val.item())

                # aggregated loss
                w_score = -cfg.W_SCORE if name in ['qb', 'to', 'wr'] else cfg.W_SCORE
                gat_loss_val = w_score * score_loss_val + cfg.W_BOUND * oob_loss_val + \
                               cfg.W_DIRE * dir_loss_val + cfg.W_VELO * velo_loss_val
                gat_loss.append(gat_loss_val.item())

                # backward
                optimizer.zero_grad()
                gat_loss_val.backward()
                optimizer.step()

                # logging
                if i % cfg.PRINT_FREQ == 0:
                    print(f'{name} epoch {e}/{epoch}, iter {i + 1}/{niters}: \n', 
                          f'   total loss={round(gat_loss_val.item(), 3)}',
                          f'   score loss={round(score_loss_val.item(), 3)}',
                          f'   oob loss={round(oob_loss_val.item(), 3)}',
                          f'   direction loss={round(dir_loss_val.item(), 3)}',
                          f'   velo loss={round(velo_loss_val.item(), 3)}\n')
                    
            # save checkpoint
            if e % cfg.SAVE_FREQ == 0:
                torch.save(gat.state_dict(), f'DeciNet/results/{cfg.NAME}/models/{name}/{name}_gat_{e}.th')
        
        # Save final model
        torch.save(gat.state_dict(), f'DeciNet/results/{cfg.NAME}/models/{name}/{name}_gat_final.th')
        
        # Summary of loss
        loss_dict = {
            'score_loss': score_loss,
            'velo_loss': velo_loss,
            'direction_loss': direction_loss,
            'oob_loss': oob_loss,
            'gat_loss': gat_loss,
        }

        # Plot loss curves
        vis_loss_curve(cfg, loss_dict, name)

        # Testing decision maker
        gat.eval()
        nums_seen = 0
        acc_gat_loss = 0
        acc_score_loss = 0
        acc_direction_loss = 0
        acc_oob_loss = 0
        acc_velo_loss = 0
        print(f'\nTesting {name} decisionmaker...')
        for i, (data, pos, v) in enumerate(test_loader):
            bsize = data.shape[0]

            # GAT forward
            gat_data = make_gat_data(data, pos, v)
            decision = gat(gat_data)
            data = make_pred_data(data, pos, decision)

            # score loss
            score_loss_val = torch.sum(ap_pred(data))
            acc_score_loss += score_loss_val.item()

            # velocity loss
            velo_loss_val = velo_loss_f(decision)
            acc_velo_loss += velo_loss_val.item() * bsize

            # direction loss
            direction_loss_val = dir_loss_f(decision, v)
            acc_direction_loss += direction_loss_val.item() * bsize

            # out of bound loss
            oob_loss_val = oob_loss_f(pos, decision)
            acc_oob_loss += oob_loss_val.item() * bsize

            # aggregated loss
            w_score = -cfg.W_SCORE if name in ['qb', 'to', 'wr'] else cfg.W_SCORE
            gat_loss_val = w_score * score_loss_val + cfg.W_BOUND * oob_loss_val + \
                           cfg.W_DIRE * dir_loss_val + cfg.W_VELO * velo_loss_val
            acc_gat_loss += gat_loss_val.item()

            nums_seen += data.shape[0]

        print(f'Offensive decision testing results: \n',
              f'   total loss={round(acc_gat_loss / nums_seen, 3)}',
              f'   score loss={round(acc_score_loss / nums_seen, 3)}',
              f'   direction loss={round(acc_direction_loss / nums_seen, 3)}',
              f'   oob loss={round(acc_oob_loss / nums_seen, 3)}',
              f'   velo loss={round(acc_velo_loss / nums_seen, 3)}\n')
