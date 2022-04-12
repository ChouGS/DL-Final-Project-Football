import numpy as np
import random

from algorithms.trajectory import GenerateTrajectory
from algorithms.eval import LoseBallProb

class Agent:
    # Basic class for Player and Ball.

    def __init__(self, x, y):
        '''
            x: real, x coordinate of the agent
            y: real, y coordinate of the agent
            ----------------------------------------------------------
            Initialization of Agent.
        '''
        # Position of the agent
        self.x = x
        self.y = y


class Player(Agent):
    distance_lb = 10    # Safety distance as in the paper
    pred_len = 10       # Length of generated trajectories
    def __init__(self, x, y, id, role):
        '''
            x: real, x coordinate of the player
            y: real, y coordinate of the player
            id: int, id of the player
            role: choose{'QB', 'WR', 'Tackle_O', 'Safety', 'Tackle_D', 'CB'}, the position of the player
            ----------------------------------------------------------
            Initialization of players.
        '''
        super(Player, self).__init__(x, y)

        # Assign player id
        self.id = id

        # The position and faction the player is in
        assert role in ['QB', 'Tackle_O', 'WR', 'Safety', 'Tackle_D', 'CB']
        self.isoffender = True if role in ['QB', 'Tackle_O', 'WR'] else False
        self.role = role
        
        # Set basic speed according to position
        if self.role == 'CB' or self.role == 'WR' or self.role == 'Safety':
            self.speed = 200
        if self.role == 'QB':
            self.speed = 120
        if self.role == 'Tackle_O' or self.role == 'Tackle_D':
            self.speed = 90

        # The color used to display this player on the gameyard
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
        '''
            holding: bool, incoming holding status
            ball: Ball instance, the ball to be held
            ----------------------------------------------------
            Switches whether the player is holding the ball 
        '''

        assert self.isoffender
        self.holding = holding
        self.color = [0x2a, 0x2a, 0xa5] if self.holding else [0x00, 0x99, 0xff]
        if self.holding:
            self.ball = ball
        else:
            self.ball = None
        
    def trajgen(self, player_list, N, mode, control_pattern):
        '''       
            player_list: list of Player instances, the information of all players
            N: int, the number of preset destinations used
            mode: choose{'1v1', 'mv1'}, the stage of the game
            control_pattern: choose{'H', 'L'}, the hyper-parameter CP
            -----------------------------------------------------------------------
            Generate all N candidate trajectories for this player.
            Details of trajectories are affected by mode, CP, and opponents in the way
        '''

        # If the player is waiting for a pass, stand still
        if self.standby:
            return np.repeat(self.trajectory.reshape(1, 4, -1), N, axis=0)

        # Decide acceleration penalty according to the position of the nearest opponent and game mode
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
        acceleration = self.speed / 3 if mode == '1v1' or self.isoffender else self.speed / 2

        # Generate virtual destinations
        r = 700                     # distance of the dests
        G = np.zeros((N, 2))        # coordinate of the dests

        # Offenders have more candidates pointing rightward
        if self.isoffender:
            for i in range(-4, 5):     
                G[(i + N) % N, 0] = r * np.cos(i * 2 * np.pi / 16) + self.x
                G[(i + N) % N, 1] = r * np.sin(i * 2 * np.pi / 16) + self.y
            for i in range(5, 8):
                G[i, 0] = r * np.cos((i * 2 - 4) * 2 * np.pi / 16) + self.x
                G[i, 1] = r * np.sin((i * 2 - 4) * 2 * np.pi / 16) + self.y
        else:
            # Defenders in 1v1 have trajectories evenly distributed
            if mode == '1v1':
                for i in range(N):
                    G[i, 0] = r * np.cos(i * 2 * np.pi / N) + self.x
                    G[i, 1] = r * np.sin(i * 2 * np.pi / N) + self.y
            # Defenders in mv1 have their trajectories pointing ahead of the ball holder
            else:                
                try:
                    # Calculate the distance to the ball holder
                    for j in range(len(player_list)):
                        if player_list[j].holding:
                            nbclass = j
                    dist_nbclass = ((player_list[nbclass].x - self.x) ** 2 + (player_list[nbclass].y - self.y) ** 2) ** 0.5
                    
                    # Distance higher than 100: go more ahead of the ball holder
                    if dist_nbclass > 100:
                        for i in range(N):
                            G[i, 0] = player_list[nbclass].x + 20 + i * 10
                            
                            if player_list[nbclass].y - self.y >= 0:
                                G[i, 1] = player_list[nbclass].y + 100 
                            else:
                                G[i, 1] = player_list[nbclass].y - 100 

                    # Distance lower than 100: go a little bit ahead of the ball holder
                    else:
                        for i in range(N):
                            G[i, 0] = player_list[nbclass].x + 20 + (i * 5) * dist_nbclass / 100
                            if player_list[nbclass].y - self.y >= 0:
                                G[i, 1] = player_list[nbclass].y + 100
                            else:
                                G[i, 1] = player_list[nbclass].y - 100

                except:
                    # If the ball is midair (no ball holder), act the same as in 1v1 mode
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
        '''
            ---------------------------------------------------------
            Move the player for 1 time step according to his trajectory.
        '''

        # If waiting for a pass, no movements
        if self.standby:
            return

        # Dequeue the first upcoming position in trajectory and modify current position
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
        '''
            x: real, the distance between a defender and the QB
            ------------------------------------------------
            Returns the passing willingness according to x.
            Don't ask why these coefficients. It's quite muthical!
        '''

        if x > 100:
            return 0
        return -91 / 45 * (6 / 455 * x + 7 / 13 + 1 / (6 / 455 * x + 7 / 13)) + 218 / 45

    def pass_or_not(self, player_list, passing_pattern):
        '''
            player_list: list of Player instances, the information of all players
            passing_pattern: choose{'H', 'L'}, the hyper-parameter PP
            ------------------------------------------------------------------------
            Decide whether the QB should pass the ball. Decisions are based on:
            - The distance to nearest opponent
            - The status of wide receivers
        '''

        assert self.role == 'QB'

        # Set coefficients according to PP
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

        # Calculate distance to nearest opponent
        min_dist_self_def = 1e5
        for player in player_list:
            if not player.isoffender:
                dist = ((player.x - self.x) ** 2 + (player.y - self.y) ** 2) ** 0.5
                min_dist_self_def = min(min_dist_self_def, dist)

        # Calculate the passing willingness
        passing_willness = self.magic_func(min_dist_self_def)

        # Inspect the status of WR's
        WR_x = []
        WR_dist_def = []
        WR_dist_QB = []
        WR_id = []
        WR_score = []

        for player in player_list:
            if player.role != 'WR':
                continue
            WR_id.append(player.id)

            # current x coordinate of the WR
            WR_x.append(player.x)

            # distance between WR and nearest defender
            min_dist_def = 1e5
            for p in player_list:
                if p.isoffender:
                    continue
                dist = ((player.x - p.x) ** 2 + (player.y - p.y) ** 2) ** 0.5
                min_dist_def = min(min_dist_def, dist)
            WR_dist_def.append(min_dist_def)

            # distance between WR and QB
            WR_dist_QB.append(((player.x - self.x) ** 2 + (player.y - self.y) ** 2) ** 0.5)

            # An overall score for that WR
            WR_score.append(alpha * WR_dist_def[-1] + beta * WR_x[-1] - gamma * WR_dist_QB[-1])
        
        # No WR (only possible if only 1 player on each team)
        if len(WR_dist_def) == 0:
            return False, None

        # The id of the wide receiver to pass to
        chosen_WR = np.argmax(np.array(WR_score))

        # Decision making
        decision = -eta * (0.8 - passing_willness) + WR_score[chosen_WR]
        if decision > 0:
            # Pass
            return True, WR_id[chosen_WR], min_dist_self_def
        else:
            # No pass
            return False, None, None
            
    def ball_pass(self, ball, target_player):
        '''
            ball: Ball instance, the ball being held by QB
            target_player: Player instance, the player to receive
            ------------------------------------------------------------- 
            Pass the ball to another player
        '''

        assert self.holding
        ball.setoff(target_player)
        self.mod_holding_state(False)
        self.ball = None

    def freeze(self):
        '''
            ---------------------------------------------------------
            If the player is being passed a ball, 
            he must immediately clean his current trajectory to stand still
        '''
        
        self.standby = True
        self.trajectory = np.repeat(np.array([self.x, self.y, 0, 0]).reshape(4, 1), 200, axis=1)

    def receive(self, ball, player_list, tol = 20):
        '''
            ball: Ball instance, the ball midair
            player_list: list of Player instances, the information of all players
            tol: real, the maximum range in which the player can try to catch the ball
            ------------------------------------------------------------------------
            Returns True/False for whether successfully receive the passing according to following logic:
            - If the ball is not midair: returns True.
            - Else, try to catch the ball. 
                If it is farther than {tol}, returns True
                If it comes closer than {tol}, perform a probability check.
                The probability will be based on the minimal distance between this player and the nearest opponent.
                Returns the outcome of this check.
        '''

        # If the ball is not thrown yet, exit with success
        if not self.standby:
            return True

        # Try to catch the ball if its close enough
        if np.sqrt((self.x - ball.x) ** 2 + (self.y - ball.y) ** 2) <= tol:
            # Calculate the distance from nearest opponent
            min_dist_self_def = 1e5
            for player in player_list:
                if not player.isoffender:
                    dist = ((player.x - self.x) ** 2 + (player.y - self.y) ** 2) ** 0.5
                    min_dist_self_def = min(min_dist_self_def, dist)

            # Probability check for losing the ball
            lose_prob = LoseBallProb(min_dist_self_def)
            p = random.random()
            if p <= lose_prob:
                # Receiving failure, defensive win
                return False
            
            # Receiving success, modify the status of this player and the ball
            self.mod_holding_state(True, ball)
            ball.mod_status('held')
            self.standby = False

            return True

        return True
        

class Ball(Agent):
    def __init__(self, x, y):
        '''
            x: real, x coordinate of the ball
            y: real, y coordinate of the ball
            ----------------------------------------------------------
            Initialization of the ball.
        '''

        super(Ball, self).__init__(x, y)

        # The status of the ball
        #   - unallocated: the ball hasn't been thrown into the gameyard
        #   - held: the ball is being held by an offensive player
        #   - midair: the ball is on its way to another offensive player
        self.status = 'unallocated'
        self.color = [0x00, 0x00, 0x00]
        self.speed = 25
        # The upcoming trajectory of the ball
        # Can be a 2*N numpy.array or some better structures
        self.trajectory = None
    
    def mod_status(self, status):
        '''
            status: choose{'unallocated', 'held', 'midair'}, the incoming status
            --------------------------------------------------------------------
            Switch the status of theball
        '''

        assert status in ['unallocated', 'held', 'midair']
        self.status = status

    def setoff(self, target_player):
        '''
            target_player: Player instance, the player to receive the passing
            --------------------------------------------------------------------
            Only called when Player.ball_pass is called.
            Modify the trajectory of the ball towards the receiver.
        '''

        self.mod_status('midair')

        dt = int(round(((self.x - target_player.x) ** 2 + (self.y - target_player.y) ** 2) ** 0.5 / self.speed))
        x_traj = np.linspace(self.x, target_player.x, dt)
        y_traj = np.linspace(self.y, target_player.y, dt)
        self.trajectory = np.block([[x_traj], [y_traj]])

    def motion(self):
        # The ball moves on itself only when it is midair
        if self.status == 'midair':
            self.x = self.trajectory[0, 0]
            self.y = self.trajectory[1, 0]
            self.trajectory = self.trajectory[:, 1:]
            if self.trajectory.shape[1] == 0:
                self.trajectory = None
