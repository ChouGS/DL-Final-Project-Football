import numpy as np

from objects.gameyard import Gameyard
from objects.agents import Player, Ball
from algorithms.eval import EvaluateTrajectories, ChooseAction
from algorithms.LCPs import LCP_lemke_howson

# Initialize game and timer
N = 16              # number of candidates

num_players = 11
tick = 0
game = Gameyard(players=num_players)
game.players[0].mod_holding_state(True, game.ball)

# Game starts
while (True):
    # Players make their decisions every 3 time steps
    if tick % 3 == 1:
        receipient_id = None

        # Generate trajectory proposals
        traj_list = []
        for player in game.players:
            traj_list.append(player.trajgen(game.players, N))

        # Decide a trajectory for each player as a Nash equilibrium

        # A: nplayer * nplayer * N * N, cost matrix between every pair of players
        A = np.zeros((2*num_players, 2*num_players, N, N))
        
        # Calculate cost matrix
        # TODO: Plug in strategy for safety
        for p in range(2*num_players):
            for q in range(2*num_players):
                if game.players[p].isoffender == game.players[q].isoffender:
                    continue
                for i in range(N):
                    for j in range(N):
                        if game.players[p].isoffender:
                            A[p, q, i, j] = EvaluateTrajectories(traj_list[p][i], traj_list[q][j],
                                                                 game.players[p].role, game.players[q].role, 
                                                                 game.players[p].x, aggressive_coef=1)
                        else:
                            A[p, q, i, j] = -EvaluateTrajectories(traj_list[p][i], traj_list[q][j], 
                                                                  game.players[p].role, game.players[q].role, 
                                                                  game.players[p].x, aggressive_coef=0)

        # Calculate probability of actions between each pair of players
        probs = np.zeros((2*num_players, N))
        for p in range(2*num_players):
            for q in range(2*num_players):
                if game.players[p].isoffender and not game.players[q].isoffender:
                    (prob1, prob2) = LCP_lemke_howson(A[p, q], A[q, p])
                    probs[p:p+1] = probs[p:p+1] + prob1.T
                    probs[q:q+1] = probs[q:q+1] + prob2.T

        # Make decision
        for p in range(2*num_players):
            prob = probs[p] / np.sum(probs[p])
            action = ChooseAction(prob)
            game.players[p].trajectory = traj_list[p][action]

    # Players and balls move every 1 time step
    # for player in game.players:
    #     player.motion()

    game.display()
    
    # Check whether the game comes to end
    # TODO: fill this up
    end, winner = game.judge_end()

    if end:
        print(f'Game over, {winner}s win.')
        break

    tick += 1
    if tick == 1000:
        break
