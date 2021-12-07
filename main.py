import numpy as np
import random
import time
import argparse

from objects.gameyard import Gameyard
from objects.agents import Player, Ball
from objects.results import Result
from algorithms.eval import EvaluateTrajectories, ChooseAction, EvaluateTrajectoriesForSafety, LoseBallProb
from algorithms.LCPs import LCP_lemke_howson


# Initialize game and timer
parser = argparse.ArgumentParser()
parser.add_argument("-ns", "--num_sims", help="[Optional] number of game simulations, 1 by default.", \
                    type=int, default=100)
parser.add_argument("-np", "--num_players", help="[Optional] number of players ON EACH TEAM, 6 by default.", \
                    type=int, default=6)
parser.add_argument("-nc", "--num_traj_cand", help="[Optional] number of candidate trajectories for each player, 12 by default.", \
                    type=int, default=12)
parser.add_argument("-op", "--offender_pattern", help="[Optional] High-level control of aggressive coef, must be 'H' or 'L'. 'L' by default.", \
                    choices=['H', 'L'], default='L')
parser.add_argument("-pp", "--passing_pattern", help="[Optional] High-level control of passing decision coef, must be 'H' or 'L'. 'L' by default.", \
                    choices=['H', 'L'], default='L')
parser.add_argument("-cp", "--control_pattern", help="[Optional] High-level control of u_penalty settings, must be 'H' or 'L'. 'L' by default.", \
                    choices=['H', 'L'], default='L')
args = parser.parse_args()

# Hyper-parameters
num_players = args.num_players
num_sims = args.num_sims
N = args.num_traj_cand
offender_pattern = args.offender_pattern
passing_pattern = args.passing_pattern
control_pattern = args.control_pattern

# Initialize result recorder
display_prefix = f"o{offender_pattern}p{passing_pattern}c{control_pattern}"
recorder = Result(f"results/{display_prefix}/results.txt")

for iter in range(num_sims):
    print(f"Simulating game {iter+1}/{num_sims} ...")
    
    # Initialize clock and game mode
    tick = 0
    mode = '1v1'
    can_pass = True

    # Create gameyard
    game = Gameyard(game_id=iter+1, players=num_players, prefix=display_prefix)
    game.players[0].mod_holding_state(True, game.ball)
    game.ball.mod_status('held')

    # Game starts
    while (True):
        # Players make their decisions every 3 time steps
        if tick % 5 == 0:
            receipient_id = None

            # Generate trajectory proposals
            traj_list = []
            for player in game.players:
                traj_list.append(player.trajgen(game.players, N, mode, control_pattern))

            # Decide a trajectory for each player as a Nash equilibrium

            # A: nplayer * nplayer * N * N, cost matrix between every pair of players
            A = np.zeros((2*num_players, 2*num_players, N, N))
            
            # Calculate cost matrix
            # TODO: Plug in strategy for safety

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
                lose_prob = LoseBallProb(min_dist_def)
                p = random.random()
                print("Ball passed!")
                print(f'pass: {round(p, 3)}, prob: {round(lose_prob, 3)}')
                if p <= lose_prob:
                    # Passing failure, defender win
                    hard_end = 'defender'
                    cause = 'passing failure'
                game.players[0].ball_pass(game.ball, game.players[receiver])
                game.players[receiver].freeze()
                mode = 'mv1'
                can_pass = False

        for player in game.players:
            if cause == '':
                rec_success = player.receive(game.ball, game.players)
                if not rec_success:
                    # Receiving failure, defender win
                    hard_end = 'defender'
                    cause = 'receiving failure'

        # Check whether the game comes to end
        # TODO: fill this up
        end, winner, cause = game.judge_end(hard_end, cause)

        if end:
            print(f'Game over, {cause}, {winner} win.')
            recorder.record(winner, cause)
            break

        game.display(tick)

        game.ball.motion()

        tick += 1
        if tick == 300:
            break
        time.sleep(0.5)

    del game

recorder.summary()