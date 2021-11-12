import numpy as np

class Agent:
    def __init__(self, x, y):
        # Position of the agent
        self.x = x
        self.y = y

class Player(Agent):
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
        
    def decision(self, player_list):
        # Make a trajectory queue using the information of the position of all players
        if self.standby:
            # Players standing by are assumed to stay where they are
            self.trajectory = np.array([self.x, self.y]).reshape(2, 1).repeat(10, 1)
        
        # TODO: Plug in tag game here
        

        # return values: 
        #     hold: whether the player decides to contuinue to hold the ball
        #     target_player: the id of the player to receive the ball
        hold = True
        target_player = None
        if not hold:
            self.ball_pass(self.ball, target_player)
        return (hold, target_player.id)

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
