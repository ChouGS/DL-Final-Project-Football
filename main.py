import numpy as np
import random
import time

from objects.gameyard import Gameyard
from objects.agents import Player, Ball
from algorithms.eval import EvaluateTrajectories, ChooseAction, EvaluateTrajectoriesForSafety, LoseBallProb
from algorithms.LCPs import LCP_lemke_howson

# Initialize game and timer
N = 12              # number of candidates

num_players = 6
tick = 0
mode = '1v1'
can_pass = True
game = Gameyard(players=num_players)
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
            traj_list.append(player.trajgen(game.players, N, mode))

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
                                                                 mode=mode, aggressive_coef=1)
                        elif game.players[p].role == 'Safety':
                            A[p, q, i, j] = EvaluateTrajectoriesForSafety(traj_list[p][i], traj_list[q][j], p, q, 
                                                                          traj_list, game.players, mode=mode)
                        else:
                            A[p, q, i, j] = EvaluateTrajectories(traj_list[p][i], traj_list[q][j], 
                                                                 game.players[p], game.players[q], 
                                                                 mode=mode, aggressive_coef=0)

        # Calculate probability of actions between each pair of players
        probs = np.zeros((2*num_players, N))
        for p in range(2*num_players):
            for q in range(2*num_players):
                if game.players[p].isoffender and not game.players[q].isoffender:
                    # if mode == '1v1':
                    (prob1, prob2) = LCP_lemke_howson(A[p, q], -A[p, q])
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
        pass_or_not, receiver, min_dist_def = game.players[0].pass_or_not(game.players)
        if pass_or_not:
            lose_prob = LoseBallProb(min_dist_def)
            p = random.random()
            print(f'pass: {round(p, 3)}, prob: {round(lose_prob, 3)}')
            if p <= lose_prob:
                # Passing failure, defensive win
                hard_end = 'defensive'
                cause = 'passing failed'
            game.players[0].ball_pass(game.ball, game.players[receiver])
            game.players[receiver].freeze()
            mode = 'mv1'
            can_pass = False

    for player in game.players:
        if cause == '':
            rec_success = player.receive(game.ball, game.players)
            if not rec_success:
                hard_end = 'defensive'
                cause = 'receiving failed'

    # Check whether the game comes to end
    # TODO: fill this up
    end, winner, cause = game.judge_end(hard_end, cause)

    if end:
        print(f'Game over, {cause}, {winner} win.')
        break

    game.display()

    game.ball.motion()

    tick += 1
    if tick == 1000:
        break
    time.sleep(0.5)
