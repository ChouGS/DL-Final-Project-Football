import numpy as np
from algorithms.LCPs import LCP_lemke_howson
from algorithms.trajectory import GenerateTrajectory

class Agent:
    def __init__(self, x, y):
        # Position of the agent
        self.x = x
        self.y = y

class Player(Agent):
    distance_lb = 10
    def __init__(self, x, y, id, role):
        super(Player, self).__init__(x, y)

        self.id = id

        # The faction the player is in
        assert role in ['offender', 'defender']
        self.role = role

        # The color used to display the player on the gameyard
        self.color = [0x00, 0x00, 0xff] if self.role == 'offender' else [0xff, 0x00, 0x00]

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
        assert self.role == 'offender'
        self.holding = holding
        self.color = [0x00, 0x99, 0xff] if self.holding else [0x00, 0x00, 0xff]
        if self.holding:
            self.ball = ball
        else:
            self.ball = None
        
    def trajgen(self, player_list, N):
        # Make a trajectory queue using the information of the position of all players
        H = 20      # predictive length
        r = 700

        # Decide acceleration penalty 
        min_distance = 1e8
        for player in player_list:
            if player.role != self.role:
                min_distance = min(min_distance, ((player.x - self.x) ** 2 + \
                                                  (player.y - self.y) ** 2) ** 0.5)
        
        u_penalty = 0.05 if min_distance > Player.distance_lb else \
                    0.05 * Player.distance_lb / min_distance

        # Generate virtual destinations
        G = np.zeros((N, 2))
        for i in range(N):
            G[i, 0] = r * np.cos(i * 2 * np.pi / (N - 1))
            G[i, 1] = r * np.sin(i * 2 * np.pi / (N - 1))

        # Set max speed for different teams
        u_max = 100 if self.role == "defender" else 125

        # Time interval
        dt = 0.1

        # Current motion state
        z_a = np.array([self.x, self.y, 0, 0])

        # Calculate trajectory candidates
        ZZ_a = np.zeros((N, 4, H))
        for n in range(N):
            g = G[n, :]
            ZZ_a[n, :, :] = GenerateTrajectory(z_a, g, H, dt, u_max, u_penalty, \
                                               (0, 500), (0, 400))  # Hard code

        return ZZ_a

    def motion(self):
        # Dequeue the first upcoming position in trajectory 
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
        self.trajectory = np.array([self.x, self.y]).reshape(2, 1).repeat(10, 1)

    def receive(self, ball):
        # Try to catch the ball if its close enough
        assert self.standby
        
        tol = 20
        if np.sqrt((self.x - ball.x) ** 2 + (self.y - ball.y) ** 2) <= tol:
            self.mod_holding_state(True, ball)
            ball.mod_status('held')
            self.standby = False


class Ball(Agent):
    def __init__(self, x, y):
        super(Ball, self).__init__(x, y)

        # The status of the ball
        #     Unallocated: the ball hasn't been thrown into the gameyard
        #     Held: the ball is being held by an offensive player
        #     Midair: the ball is on its way to another offensive player
        self.status = 'unallocated'
        self.color = [0x00, 0x00, 0x00]

        # The upcoming trajectory of the ball
        # Can be a 2*N numpy.array or some better structures
        self.trajectory = None
    
    def mod_status(self, status):
        assert status in ['unallocated', 'held', 'midair']
        self.status = status

    def setoff(self, target_player):
        self.mod_status('midair')
        # TODO: modify the trajectory of the ball to be a straight line from 
        #       the current position to the osition of the target player
        

    def motion(self):
        # The ball moves itself only when it is midair
        if self.status == 'midair':
            self.x = self.trajectory[0, 0]
            self.y = self.trajectory[1, 0]
            self.trajectory = self.trajectory[:, 1:]
            if self.trajectory.shape[1] == 0:
                self.trajectory = None
