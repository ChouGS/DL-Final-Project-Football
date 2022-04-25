import numpy as np
import random
import os
import itertools
import argparse

from objects.gameyard import Gameyard
from objects.results import Result
from algorithms.eval import EvaluateTrajectories, ChooseAction, EvaluateTrajectoriesForSafety, LoseBallProb
from algorithms.LCPs import LCP_lemke_howson
from algorithms.DL_solver import DLsolver

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Initialize game and timer
parser = argparse.ArgumentParser()
parser.add_argument("-ns", "--num_sims", help="[Optional] number of game simulations, 1 by default.", \
                    type=int, default=1)
parser.add_argument("-np", "--num_players", help="[Optional] number of players ON EACH TEAM, 6 by default.", \
                    type=int, default=6)
parser.add_argument("-nc", "--num_traj_cand", help="[Optional] number of candidate trajectories for each player, 12 by default.", \
                    type=int, default=12)
parser.add_argument("-op", "--offender_pattern", help="[Optional] High-level control of aggressive coef, must be 'H' or 'L'. 'L' by default.", \
                    choices=['H', 'L'], default='L')
parser.add_argument("-pp", "--passing_pattern", help="[Optional] High-level control of passing decision coef, must be 'H' or 'L'. 'L' by default.", \
                    choices=['H', 'L'], default='H')
parser.add_argument("-cp", "--control_pattern", help="[Optional] High-level control of u_penalty settings, must be 'H' or 'L'. 'L' by default.", \
                    choices=['H', 'L'], default='L')
parser.add_argument("-oa", "--offensive_agent", help="[Optional] Decision agent for offenders, must be 'CGT' or 'DL'.", \
                    choices=['CGT', 'DL'], default='CGT')
parser.add_argument("-da", "--defensive_agent", help="[Optional] Decision agent for defenders, must be 'CGT' or 'DL'.", \
                    choices=['CGT', 'DL'], default='CGT')
parser.add_argument("-opath", "--offensive_agent_path", help="[Optional] DL model path for offenders.", \
                    type=str, default='algorithms/DL_model/off_gat_final.th')
parser.add_argument("-dpath", "--defensive_agent_path", help="[Optional] DL model path for defenders.", \
                    type=str, default='algorithms/DL_model/def_gat_final.th')
parser.add_argument("-cpath", "--config_path", help="[Optional] Configuration path for initializing DL model.", \
                    type=str, default='algorithms/DL_model/config/gat_small_att.yaml')
parser.add_argument("-l", "--log", help="Whether to write out log files", default=True)
parser.add_argument("-v", "--video", help="Whether to write out videos", default=True)
parser.add_argument("-d", "--gen_data", help="Whether to generate labeled data", default=False)
args = parser.parse_args()

# Hyper-parameters
num_players = args.num_players              # Number of players on each team
num_sims = args.num_sims                    # The number of simulations (separate games) to be simulated
N = args.num_traj_cand                      # The number of candidate trajectories for each player
offender_pattern = args.offender_pattern    # The OP settings
passing_pattern = args.passing_pattern      # The PP settings
control_pattern = args.control_pattern      # The CP settings
logging = args.log
video = args.video
gen_data = args.gen_data

# Initialize result recorder
display_prefix = f"o{offender_pattern}p{passing_pattern}c{control_pattern}"
if logging:
    recorder = Result(f"results/{display_prefix}/results.txt")

# Initialize DL model if DL method is used
off_solver = None
def_solver = None
if args.offensive_agent == 'DL':
    off_solver = DLsolver(args.offensive_agent_path, args.config_path)
if args.defensive_agent == 'DL':
    def_solver = DLsolver(args.defensive_agent_path, args.config_path)

# Data collector
data_root = f'raw_data/{num_players}{display_prefix}'
os.makedirs(os.path.join(data_root, 'after_passing'), exist_ok=True)
os.makedirs(os.path.join(data_root, 'before_passing'), exist_ok=True)
data_cnt_ap = 0
data_cnt_bp = 0
existing_num = 0
for fname in os.listdir(os.path.join(data_root, 'after_passing')):
    existing_num = max(existing_num, int(fname.split('.')[0]) + 1)

# Simulation starts...
for iter in range(num_sims):
    print(f"Simulating game {iter+1}/{num_sims} ...")
    if gen_data:
        data_bp = None
        data_ap = None

    # Initialize clock and game mode
    tick = 0
    mode = '1v1'
    can_pass = True

    # Create gameyard
    game = Gameyard(game_id=iter+1, prefix=display_prefix, n_traj=N, video=video, players=num_players)
    game.players[0].mod_holding_state(True, game.ball)
    game.ball.mod_status('held')

    # Game starts
    while (True):
        # Players make their decisions every 3 time steps
        if tick % 5 == 0:
            receipient_id = None

            # Generate trajectory proposals
            traj_list = []
            ball_pos = np.array([[game.ball.x, game.ball.y]])
            for player in game.players:
                if player.isoffender:
                    traj_list.append(player.trajgen(game.players, mode, control_pattern, ball_pos, off_solver))
                else:
                    traj_list.append(player.trajgen(game.players, mode, control_pattern, ball_pos, def_solver))

            # Decide a trajectory for each player as a Nash equilibrium

            # A: nplayer * nplayer * N * N, cost matrix between every pair of players
            # N = 1 if DL strategy is used, N = args.num_traj_cand if CGT is used
            A = np.zeros((2*num_players, 2*num_players, N, N))
            
            # Calculate cost matrix
            for p in range(2*num_players):
                if game.players[p].standby:
                    continue
                for q in range(2*num_players):
                    if game.players[p].isoffender == game.players[q].isoffender:
                        continue
                    for i in range(N):
                        for j in range(N):
                            if game.players[p].isoffender:
                                A[p, q, i, j] = EvaluateTrajectories(traj_list[p][i], traj_list[q][j],
                                                                    game.players[p], game.players[q],
                                                                    mode, offender_pattern)
                            elif game.players[p].role == 'Safety':
                                A[p, q, i, j] = EvaluateTrajectoriesForSafety(traj_list[p][i], traj_list[q][j], p, q, 
                                                                              traj_list, game.players, mode)
                            else:
                                A[p, q, i, j] = EvaluateTrajectories(traj_list[p][i], traj_list[q][j], 
                                                                    game.players[p], game.players[q], 
                                                                    mode, offender_pattern)

            # Calculate probability of actions between each pair of players
            probs = np.zeros((2*num_players, N))
            for p in range(2*num_players):
                for q in range(2*num_players):
                    if game.players[p].isoffender != game.players[q].isoffender:
                        # if mode == '1v1':
                        (prob1, prob1_2) = LCP_lemke_howson(A[p, q], -A[p, q])
                        (prob2, prob2_2) = LCP_lemke_howson(A[q, p], -A[q, p])
                        # else:
                        #     (prob1, prob2) = LCP_lemke_howson(A[p, q], A[q, p].T)
                        probs[p:p+1] = probs[p:p+1] + prob1.T
                        probs[q:q+1] = probs[q:q+1] + prob2.T

            # Make decision
            for p in range(2*num_players):
                prob = probs[p] / np.sum(probs[p])
                action = ChooseAction(prob)
                game.players[p].trajectory = traj_list[p][action]

            # Generate data
            if gen_data:
                # IMPORTANT: Data structure (11 player game as example)
                # 0-2: (x, y, dist) of teammate 1, ... till 30-32
                # 33-35: (x, y, dist) of rival 1, ..., till 63-65
                # 66-68: (x, y, dist) of ball
                # 69-70: (x, y) of self
                # 71-72: (vx, vy) of last tick
                # 73-76: self next decision (dx, dy, vx, vy)
                # 77: final x label
                # 78: score label (To be added via postprocessing)
                # 79: touchdown or not
                players_xy = [[game.players[i].x, game.players[i].y] for i in range(2 * num_players)]
                pxy_perm = list(itertools.permutations(players_xy, 2))

                for i in range(2 * num_players):
                    pxy_perm.insert((2 * num_players + 1) * i, ([0, 0], [0, 0]))

                players_xy = np.array(players_xy)
                pxy_perm_1 = np.array([pxy_perm[i][0] for i in range(len(pxy_perm))])   # (N * N) * 2
                pxy_perm_2 = np.array([pxy_perm[i][1] for i in range(len(pxy_perm))])   # (N * N) * 2
                dist_flatten = np.sqrt(np.sum(np.power(pxy_perm_1 - pxy_perm_2, 2), 1))
                dist_matrix = dist_flatten.reshape((2 * num_players, 2 * num_players))
                dist_matrix[:num_players, :num_players] *= -1
                dist_matrix[num_players:2*num_players, num_players:2*num_players] *= -1

                new_data_item = np.zeros((2 * num_players, 3 * (2 * num_players) + 13))
                for i in range(2 * num_players):
                    players_idx = list(range(2 * num_players))
                    new_data_item[i, 0:3*(2*num_players):3] = players_xy[players_idx, 0]
                    new_data_item[i, 1:3*(2*num_players):3] = players_xy[players_idx, 1]
                    new_data_item[i, 2:3*(2*num_players):3] = dist_matrix[i, players_idx]
                    new_data_item[i, 3*(2*num_players)] = game.ball.x
                    new_data_item[i, 3*(2*num_players) + 1] = game.ball.y
                    new_data_item[i, 3*(2*num_players) + 2] = np.sqrt((game.ball.y - game.players[i].y) ** 2 + (game.ball.x - game.players[i].x) ** 2)
                    new_data_item[i, 3*(2*num_players) + 2] *= -1 if i < num_players else 1
                    new_data_item[i, 3*(2*num_players) + 3:3*(2*num_players) + 5] = players_xy[i]
                    new_data_item[i, 3*(2*num_players) + 7:3*(2*num_players) + 11] = game.players[i].trajectory[:, 0]            
                    if can_pass:
                        if data_bp is None:
                            new_data_item[i, 3*(2*num_players) + 5:3*(2*num_players) + 7] = \
                                new_data_item[i, 3*(2*num_players) + 9:3*(2*num_players) + 11]
                        else:
                            new_data_item[i, 3*(2*num_players) + 5:3*(2*num_players) + 7] = \
                                data_bp[i - 2*num_players, 3*(2*num_players) + 9:3*(2*num_players) + 11]
                    else:
                        if data_ap is None:
                            new_data_item[i, 3*(2*num_players) + 5:3*(2*num_players) + 7] = \
                                new_data_item[i, 3*(2*num_players) + 9:3*(2*num_players) + 11]
                        else:
                            new_data_item[i, 3*(2*num_players) + 5:3*(2*num_players) + 7] = \
                                data_ap[i - 2*num_players, 3*(2*num_players) + 9:3*(2*num_players) + 11]

                if can_pass:
                    if data_bp is None:
                        data_bp = new_data_item
                    else:
                        data_bp = np.concatenate([data_bp, new_data_item], 0)
                else:
                    if data_ap is None:
                        data_ap = new_data_item
                    else:
                        data_ap = np.concatenate([data_ap, new_data_item], 0)

        # Players and balls move every 1 time step
        for player in game.players:
            player.motion()

        if game.players[0].x > Gameyard.start_line_x:
            can_pass = False
            mode = 'mv1'

        # If the offenders still have the right to pass the ball, judge whether they should do so
        hard_end = None    
        cause = ''
        if can_pass:
            pass_or_not, receiver, min_dist_def = game.players[0].pass_or_not(game.players, passing_pattern)
            if pass_or_not:
                # Probability check for ballpass failure
                lose_prob = LoseBallProb(min_dist_def)
                p = random.random()
                print("Ball passed!")
                # print(f'pass: {round(p, 3)}, prob: {round(lose_prob, 3)}')
                if p <= lose_prob:
                    # Passing failure, defender win
                    hard_end = 'defender'
                    cause = 'passing failure'
                game.players[0].ball_pass(game.ball, game.players[receiver])
                game.players[receiver].freeze()
                mode = 'mv1'
                can_pass = False

        # If the passing is done successfully, the receiver tries to catch the ball
        for player in game.players:
            if cause == '':
                rec_success = player.receive(game.ball, game.players)
                if not rec_success:
                    # Receiving failure, defender win
                    hard_end = 'defender'
                    cause = 'receiving failure'

        # Check whether the game comes to end. If so, end with results
        end, winner, cause = game.judge_end(hard_end, cause)
        if end:
            print(f'Game over, {cause}, {winner} win.')
            if logging:
                recorder.record(winner, cause)
            if gen_data:
                data_bp[:, -1] = game.ball.x
                data_ap[:, -1] = game.ball.x
                data_cnt_ap += data_ap.shape[0]
                data_cnt_bp += data_bp.shape[0]
                np.save(os.path.join(data_root, 'after_passing', f'{iter+existing_num}.npy'), data_ap)
                np.save(os.path.join(data_root, 'before_passing', f'{iter+existing_num}.npy'), data_bp)
                print(f'Simulation {iter+1}/{num_sims} done.')
                print(f'After passing data +{data_ap.shape[0]}, total {data_cnt_ap}')
                print(f'Before passing data +{data_bp.shape[0]}, total {data_cnt_bp}')
            break

        # Do visualization
        if video:
            game.display(tick)

        # Ball motion if it is midair
        game.ball.motion()

        # Time lapse
        tick += 1
        if tick == 300:
            if gen_data:
                data_bp[:, -1] = game.ball.x
                data_ap[:, -1] = game.ball.x
                data_cnt_ap += data_ap.shape[0]
                data_cnt_bp += data_bp.shape[0]
                np.save(os.path.join(data_root, 'after_passing', f'{iter+existing_num}.npy'), data_ap)
                np.save(os.path.join(data_root, 'before_passing', f'{iter+existing_num}.npy'), data_bp)
                print(f'After passing data +{data_ap.shape[0]}, total {data_cnt_ap}')
                print(f'Before passing data +{data_bp.shape[0]}, total {data_cnt_bp}')
            break
        # time.sleep(0.5)

    # Make video
    if video:
        game.vw.release()
    
    # Clean the previous game
    del game

# Store the statistics for simulation result
if logging:
    recorder.summary()

# Combine data and visualize distribution
if gen_data:
    import make_data
    make_data.make_data(f'{num_players}{display_prefix}')
    import vis_data_distribution
    vis_data_distribution.vis_distribution()
