from cv2 import OPTFLOW_FARNEBACK_GAUSSIAN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import os
import re
import shutil
import argparse

from pred_model import PredAE, PredX, PredG, PredATT
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
os.makedirs(f'GameNet/results/{cfg.NAME}/config', exist_ok=True)
os.makedirs(f'GameNet/results/{cfg.NAME}/models', exist_ok=True)
shutil.copy(args.config, f'GameNet/results/{cfg.NAME}/config/{args.config.split("/")[-1]}')

def generateNew(stats, pred):
    orig_data = stats[:,:69]
    att_feature = orig_data.reshape(-1, 23, 3)
    att_feature = torch.cat([att_feature, torch.zeros(stats.shape[0], 23, 3)], 2)
    vel = stats[:,69:71].unsqueeze(1).repeat(1,23,1)
    att_feature[:, :, 2:4] = vel
    att_feature[:, :, 4:] = pred.unsqueeze(1).repeat(1,23,1)
    
    return att_feature

if __name__ == '__main__':
    # Load data
    ap_data = np.load(os.path.join(cfg.DATA_DIR, 'data_after_passing.npy'))
    subject_num = ap_data.shape[0] // 22
    print(ap_data.shape[0])
    offensive_data = []
    defensive_data = []
    for i in range(subject_num):
        for kk in range(11):
            offensive_data.append(ap_data[kk+22*i, :])
            defensive_data.append(ap_data[kk+11+22*i, :])
    offensive_data = np.array(offensive_data)
    defensive_data = np.array(defensive_data)
        

    if cfg.PRED == 'MLP':
        ap_train_op = PredictorDataset(offensive_data[:int(round(offensive_data.shape[0] * (1 - cfg.DATA.TESTRATIO)))])
        ap_train_dp = PredictorDataset(defensive_data[:int(round(defensive_data.shape[0] * (1 - cfg.DATA.TESTRATIO)))])

       # ap_test = PredictorDataset(ap_data[int(round(ap_data.shape[0] * (1 - cfg.DATA.TESTRATIO))):])
    elif cfg.PRED == 'ATT':
        ap_train = AttentionDataset(ap_data[:int(round(ap_data.shape[0] * (1 - cfg.DATA.TESTRATIO)))])
        ap_test = AttentionDataset(ap_data[int(round(ap_data.shape[0] * (1 - cfg.DATA.TESTRATIO))):])
    else:
        raise NotImplementedError(f'Not supported predictor type {cfg.PRED}')

    ap_trloader_op = DataLoader(ap_train_op, batch_size=cfg.DATA.TRAINBS, shuffle=True)
    ap_trloader_dp = DataLoader(ap_train_dp, batch_size=cfg.DATA.TRAINBS, shuffle=True)

    # ap_teloader = DataLoader(ap_test, batch_size=cfg.DATA.TESTBS, shuffle=True)


    if cfg.USE_AE:
        cfg.MODEL.AE.IN_DIM = cfg.DATA.FDIM

    cfg.freeze()

    # Train Offesnvie team : maximize the final score
    cos_loss = nn.CosineSimilarity(dim=1, eps=1e-6)
    if cfg.PRED == 'MLP': 
        ap_pred = PredG(cfg.MODEL.G)
        ap_pred_opt = optim.Adam(ap_pred.parameters(), lr=cfg.MODEL.G.LR)
        epoch = cfg.MODEL.G.EPOCH
        ap_eval = PredATT(cfg.MODEL.ATT)
        #with open("ap_pred.th", 'rb') as f:
        state_dict = torch.load('/Users/jacksonwang/Downloads/ap_pred_final_small.th')
        ap_eval.load_state_dict(state_dict)
        ap_eval.eval()
        # ap_eval_opt = optim.Adam(ap_eval.parameters(), lr=cfg.MODEL.X.LR)

    elif cfg.PRED == 'ATT':
        ap_pred = PredATT(cfg.MODEL.ATT)
        ap_pred_opt = optim.Adam(ap_pred.parameters(), lr=cfg.MODEL.ATT.LR)
        epoch = cfg.MODEL.ATT.EPOCH
    
    ap_pred_loss = []

    print('\nTraining after_passing predictor...')
    niters = len(ap_trloader_op) // cfg.DATA.TRAINBS + 1
    for e in range(epoch):
        for i, (data, velocity, orig_data, _) in enumerate(ap_trloader_op):

            velocity_pred = ap_pred(data)
            data_ref = generateNew(data, velocity_pred)
            scores = ap_eval(data_ref)
            sim_loss = torch.mean(1/(1.1+cos_loss(velocity_pred, velocity)))
            norm_loss = torch.mean(10 ** (torch.norm(velocity_pred,dim=1)-10))
            sco_loss = torch.mean(scores)
            loss_val = sim_loss - sco_loss + 100 * norm_loss

            ap_pred_loss.append(loss_val.item())

            ap_pred_opt.zero_grad()
            loss_val.backward()
            ap_pred_opt.step()

            if i % cfg.PRINT_FREQ == 0:
                print(f'Pred epoch {e}/{epoch}, iter {i + 1}/{niters}: ap_pred_loss={loss_val} cos loss={sim_loss} score loss={sco_loss} norm loss={norm_loss}')
    
    loss_dict = {
        'ap_x_loss': ap_pred_loss,
    }


    # Plot loss curves
    vis_loss_curve(cfg, loss_dict)

    
    os.makedirs(f'PredNet/results/{cfg.NAME}/models', exist_ok=True)
    torch.save(bp_ae.state_dict(), f'PredNet/results/{cfg.NAME}/models/bp_ae.th')
    torch.save(ap_pred.state_dict(), f'PredNet/results/{cfg.NAME}/models/ap_pred.th')
    torch.save(bp_pred.state_dict(), f'PredNet/results/{cfg.NAME}/models/bp_pred.th')
    