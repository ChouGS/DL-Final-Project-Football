# DL-Final-Project-Football

Run `objects/gameyard.py` to see several sample visualizations

## Quick Start Guide

### To run new simulations

`python main.py`

commandline flags:

`-ns` or `--num_sims` indicates the number of simulations.

`-np` or `--num_players` indicates the number of players on each team.

`-oa` or `--offensive_agent` should be set to 'DL' if deep learning agent is used on the offensive team.

`-da` or `--defensive_agent` should be set to 'DL' if deep learning agent is used on the defensive team.

To run the code successfully you should have the following files:

`algorithms/DL_model/config/gat_small_att.yaml` as the game configuration
`algorithms/DL_model/{xx}_gat_final.th` as the GameNet checkpoint for different positions. Here {xx}=qb/to/wr/td/sf/cb.

### To train ScoreNet

`python ./ScoreNet/train_ap_att.py`

To run the code successfully you should have the following files:

`ScoreNet/configs/____.yaml` as the configuration used for ScoreNet training.

`raw_data/11oLpHcL/synthesized/data_after_passing.npy` as the input data file.

After training is done you can see `ScoreNet/results` for results.

### To train GameNet

`python ./GameNet/train_ap_att.py`

To run the code successfully you should have the following files:

`GameNet/configs/____.yaml` as the configuration used for GameNet training.

`raw_data/11oLpHcL/synthesized/data_after_passing_{xx}.npy` as the input data file. Here {xx}=qb/to/wr/td/sf/cb.

`GameNet/pretrained/ap_pred_final.th` as the pretrained weight of ScoreNet.

After training is done you can see `GameNet/results` for results.
