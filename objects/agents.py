import numpy as np
import random

from algorithms.LCPs import LCP_lemke_howson
from algorithms.trajectory import GenerateTrajectory
from algorithms.eval import LoseBallProb

class Agent:
    def __init__(self, x, y):
        # Position of the agent
        self.x = x
        self.y = y

class Player(Agent):
    distance_lb = 10
    pred_len = 10
    def __init__(self, x, y, id, role):
        super(Player, self).__init__(x, y)

        self.id = id

        # The faction the player is in
        assert role in ['QB', 'Tackle_O', 'WR', 'Safety', 'Tackle_D', 'CB']
        self.isoffender = True if role in ['QB', 'Tackle_O', 'WR'] else False
        self.role = role
        if self.role == 'CB' or self.role == 'WR' or self.role == 'Safety':
            self.speed = 200
        if self.role == 'QB':
            self.speed = 120
        if self.role == 'Tackle_O' or self.role == 'Tackle_D':
            self.speed = 90

        # The color used to display the player on the gameyard
        self.color = [0x00, 0x00, 0xff] if self.isoffender else [0xff, 0x00, 0x00]
        if self.role == 'WR':
            self.color = [0xb4, 0x69, 0xff]
        if self.role == 'Safety':
            self.color = [0xff, 0xbf, 0x00]
        if self.role == 'QB':
            self.color = [0x00, 0x99, 0xff]

        # Whether this player is holding the ball
        self.holding = False
        self.ball = None
        
        # Whether this player needs to stand still to receive a pass
        self.standby = False

        # The upcoming trajectory of the player
        # Can be a 2*N numpy.array or some better structures
        self.trajectory = None

    def mod_holding_state(self, holding, ball=None):
        # Switches whether the player is holding the ball 
        assert self.isoffender
        self.holding = holding
        self.color = [0x2a, 0x2a, 0xa5] if self.holding else [0x00, 0x99, 0xff]
        if self.holding:
            self.ball = ball
        else:
            self.ball = None
        
    def trajgen(self, player_list, N, mode, control_pattern):
        breaking = False
        if self.standby:
            return np.repeat(self.trajectory.reshape(1, 4, -1), N, axis=0)

        # Make a trajectory queue using the information of the position of all players
        r = 700

        # Decide acceleration penalty 
        min_distance = 1e8
        closest_role = None
        for player in player_list:
            if player.isoffender != self.isoffender:
                min_distance = min(min_distance, ((player.x - self.x) ** 2 + \
                                                  (player.y - self.y) ** 2) ** 0.5)
                closest_role = player.role
        
        u_penalty = 0.1
        if control_pattern == 'L':
            if mode == '1v1':
                if min_distance < Player.distance_lb:
                    if closest_role == 'Tackle_D' or closest_role == 'Tackle_O':
                        u_penalty = 0.6 * Player.distance_lb / min_distance
                    elif closest_role == 'CB' or closest_role == 'Safety':
                        u_penalty = 0.3 * Player.distance_lb / min_distance
                    else:
                        u_penalty = 0.2 * Player.distance_lb / min_distance
            if mode == 'mv1':
                if min_distance < Player.distance_lb:
                    if closest_role == 'Tackle_D' or closest_role == 'Tackle_O':
                        u_penalty = 0.6 * Player.distance_lb / min_distance
                    elif closest_role == 'CB' or closest_role == 'Safety':
                        u_penalty = 0.4 * Player.distance_lb / min_distance
                    else:
                        u_penalty = 0.1 * Player.distance_lb / min_distance

        elif control_pattern == 'H':
            if mode == '1v1':
                if min_distance < Player.distance_lb:
                    if closest_role == 'Tackle_D' or closest_role == 'Tackle_O':
                        u_penalty = 1.2 * Player.distance_lb / min_distance
                    elif closest_role == 'CB' or closest_role == 'Safety':
                        u_penalty = 0.6 * Player.distance_lb / min_distance
                    else:
                        u_penalty = 0.4 * Player.distance_lb / min_distance
            if mode == 'mv1':
                if min_distance < Player.distance_lb:
                    if closest_role == 'Tackle_D' or closest_role == 'Tackle_O':
                        u_penalty = 1.2 * Player.distance_lb / min_distance
                    elif closest_role == 'CB' or closest_role == 'Safety':
                        u_penalty = 0.8 * Player.distance_lb / min_distance
                    else:
                        u_penalty = 0.2 * Player.distance_lb / min_distance
        else:
            raise ValueError("control_pattern must be either 'H' or 'L'.")
            
        # Set v_bound
        v_bound = self.speed * np.sin(np.pi / (2 * Player.distance_lb) * min_distance) \
                    if min_distance < Player.distance_lb else self.speed
        # if mode == 'mv1' and not self.isoffender:
        #     v_bound *= 2
        acceleration = self.speed / 3 if mode == '1v1' or self.isoffender else self.speed / 2

        # Generate virtual destinations
        G = np.zeros((N, 2))
        if self.isoffender:
            for i in range(-4, 5):     
                G[(i + N) % N, 0] = r * np.cos(i * 2 * np.pi / 16) + self.x
                G[(i + N) % N, 1] = r * np.sin(i * 2 * np.pi / 16) + self.y
            
            for i in range(5, 8):
                G[i, 0] = r * np.cos((i * 2 - 4) * 2 * np.pi / 16) + self.x
                G[i, 1] = r * np.sin((i * 2 - 4) * 2 * np.pi / 16) + self.y
        else:
            if mode == '1v1':
                for i in range(N):
                    G[i, 0] = r * np.cos(i * 2 * np.pi / N) + self.x
                    G[i, 1] = r * np.sin(i * 2 * np.pi / N) + self.y
            else:                
                try:
                    for j in range(len(player_list)):
                        if player_list[j].holding:
                            nbclass = j
                    dist_nbclass = ((player_list[nbclass].x-self.x) ** 2 + (player_list[nbclass].y-self.y) ** 2) ** 0.5
                    
                    if dist_nbclass > 100:
                        for i in range(N):
                            G[i, 0] = player_list[nbclass].x + 20 + i * 10
                            
                            if player_list[nbclass].y - self.y >= 0:
                                G[i, 1] = player_list[nbclass].y + 100 #- 10 - i * 5
                            else:
                                G[i, 1] = player_list[nbclass].y - 100 #- 10 - (N - i) * 5
                            
                    else:
                        for i in range(N):
                            G[i, 0] = player_list[nbclass].x + 20 + (i * 5) * dist_nbclass / 100
                            if player_list[nbclass].y - self.y >= 0:
                                G[i, 1] = player_list[nbclass].y + 100 #- 10 - (i * 1) * dist_nbclass / 100
                            else:
                                G[i, 1] = player_list[nbclass].y - 100 #- 10 - ((N - i) * 1) * dist_nbclass / 100
                    # print('y',player_list[nbclass].y)
                    # print('ya',player_list[nbclass].y - self.y)
                    # print(G[i, 1])
                    breaking = True
                except:
                    for i in range(N):
                        G[i, 0] = r * np.cos(i * 2 * np.pi / N) + self.x
                        G[i, 1] = r * np.sin(i * 2 * np.pi / N) + self.y

        # Time interval
        dt = 0.1

        # Current motion state
        z_a = np.array([self.x, self.y, 0, 0])

        # Calculate trajectory candidates
        ZZ_a = np.zeros((N, 4, Player.pred_len))
        for n in range(N):
            g = G[n, :]
            if self.isoffender:
                ZZ_a[n, :, :] = GenerateTrajectory(z_a, g, Player.pred_len, dt, acceleration, u_penalty, \
                                                   (-10, 510), (-20, 420), v_bound)  # Hard code
            else:
                ZZ_a[n, :, :] = GenerateTrajectory(z_a, g, Player.pred_len, dt, acceleration, u_penalty, \
                                                   (-30, 530), (-50, 450), v_bound)  # Hard code

        return ZZ_a

    def motion(self):
        # Dequeue the first upcoming position in trajectory 
        if self.standby:
            return
        if self.trajectory is not None:
            self.x = self.trajectory[0, 0]
            self.y = self.trajectory[1, 0]
            # Ball moves with the player if it's held
            if self.ball is not None:
                self.ball.x = self.trajectory[0, 0]
                self.ball.y = self.trajectory[1, 0]
            self.trajectory = self.trajectory[:, 1:]
            if self.trajectory.shape[1] == 0:
                self.trajectory = None

    def magic_func(self, x):
        if x > 100:
            return 0
        return -91 / 45 * (6 / 455 * x + 7 / 13 + 1 / (6 / 455 * x + 7 / 13)) + 218 / 45

    def pass_or_not(self, player_list, passing_pattern):
        if passing_pattern == 'H':
            alpha = 50
            beta = 0.5
            gamma = 0.001
            eta = 40000
        elif passing_pattern == 'L':
            alpha = 8
            beta = 5
            gamma = 0.001
            eta = 6000
        else:
            raise ValueError("passing_pattern must be either 'H' or 'L'.")

        assert self.role == 'QB'

        min_dist_self_def = 1e5
        for player in player_list:
            if not player.isoffender:
                dist = ((player.x - self.x) ** 2 + (player.y - self.y) ** 2) ** 0.5
                min_dist_self_def = min(min_dist_self_def, dist)

        passing_willness = self.magic_func(min_dist_self_def)

        WR_x = []
        WR_dist_def = []
        WR_dist_QB = []
        WR_id = []
        WR_score = []

        for player in player_list:
            if player.role != 'WR':
                continue
            WR_id.append(player.id)
            WR_x.append(player.x)

            min_dist_def = 1e5
            for p in player_list:
                if p.isoffender:
                    continue
                dist = ((player.x - p.x) ** 2 + (player.y - p.y) ** 2) ** 0.5
                min_dist_def = min(min_dist_def, dist)
            WR_dist_def.append(min_dist_def)

            WR_dist_QB.append(((player.x - self.x) ** 2 + (player.y - self.y) ** 2) ** 0.5)

            WR_score.append(alpha * WR_dist_def[-1] + beta * WR_x[-1] - gamma * WR_dist_QB[-1])
        
        if len(WR_dist_def) == 0:
            return False, None

        chosen_WR = np.argmax(np.array(WR_score))

        # Judge no-pass
        decision = -eta * (0.8 - passing_willness) + WR_score[chosen_WR]

        if decision > 0:
            # pass
            return True, WR_id[chosen_WR], min_dist_self_def
        else:
            return False, None, None
            
    def ball_pass(self, ball, target_player):
        # Pass the ball to another player
        assert self.holding
        ball.setoff(target_player)
        self.mod_holding_state(False)
        self.ball = None

    def freeze(self):
        # If the player is being passed a ball, 
        # he must immediately abandon his current trajectory
        self.standby = True
        self.trajectory = np.repeat(np.array([self.x, self.y, 0, 0]).reshape(4, 1), 200, axis=1)

    def receive(self, ball, player_list):
        # Try to catch the ball if its close enough
        if not self.standby:
            return True

        tol = 20
        if np.sqrt((self.x - ball.x) ** 2 + (self.y - ball.y) ** 2) <= tol:
            min_dist_self_def = 1e5
            for player in player_list:
                if not player.isoffender:
                    dist = ((player.x - self.x) ** 2 + (player.y - self.y) ** 2) ** 0.5
                    min_dist_self_def = min(min_dist_self_def, dist)

            lose_prob = LoseBallProb(min_dist_self_def)
            p = random.random()
            print(f'rec: {round(p, 3)}, prob: {round(lose_prob, 3)}')
            if p <= lose_prob:
                # Passing failure, defensive win
                return False

            self.mod_holding_state(True, ball)
            ball.mod_status('held')
            self.standby = False

            return True

        return True
        

class Ball(Agent):
    def __init__(self, x, y):
        super(Ball, self).__init__(x, y)

        # The status of the ball
        #     Unallocated: the ball hasn't been thrown into the gameyard
        #     Held: the ball is being held by an offensive player
        #     Midair: the ball is on its way to another offensive player
        self.status = 'unallocated'
        self.color = [0x00, 0x00, 0x00]
        self.speed = 25
        # The upcoming trajectory of the ball
        # Can be a 2*N numpy.array or some better structures
        self.trajectory = None
    
    def mod_status(self, status):
        assert status in ['unallocated', 'held', 'midair']
        self.status = status

    def setoff(self, target_player):
        self.mod_status('midair')

        dt = int(round(((self.x - target_player.x) ** 2 + (self.y - target_player.y) ** 2) ** 0.5 / self.speed))
        x_traj = np.linspace(self.x, target_player.x, dt)
        y_traj = np.linspace(self.y, target_player.y, dt)
        self.trajectory = np.block([[x_traj], [y_traj]])

    def motion(self):
        # The ball moves itself only when it is midair
        if self.status == 'midair':
            self.x = self.trajectory[0, 0]
            self.y = self.trajectory[1, 0]
            self.trajectory = self.trajectory[:, 1:]
            if self.trajectory.shape[1] == 0:
                self.trajectory = None
