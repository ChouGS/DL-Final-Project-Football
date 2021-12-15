import numpy as np
import cv2
import os
import random
from objects.agents import Ball, Player
from matplotlib import pyplot

class Gameyard:
    
    w = 500                     # The width of the gameyard
    h = 400                     # The height of the gameyard
    defensive_line_x = 400      # The x coordinate of the touchdown line
    start_line_x = 200          # The x coordinate of the scrimmage

    def __init__(self, game_id, prefix, players=1):
        '''
            game_id: int, game ID
            prefix: str, output file name prefix
            players: int, number of players onn each team
            -------------------------------------------------------
            Initialization of gameyard.
        '''

        # Number of different positions for different {players} settings
        role_dict = {
            'QB':       [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'WR':       [-1, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            'Tackle_O': [-1, 0, 0, 1, 2, 3, 3, 4, 5, 6, 7, 8],
            'Safety':   [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
            'CB':       [-1, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            'Tackle_D': [-1, 0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7]
        }

        # Game ID. Only used for output file naming
        self.id = game_id

        # Create offenders
        self.players = [Player(125, 200, 0, 'QB')]
        for i in range(role_dict['WR'][players]):
            if i == 0:
                self.players.append(Player(175, 325, len(self.players), 'WR'))
            if i == 1:
                self.players.append(Player(175, 75, len(self.players), 'WR'))
        player_y = np.linspace(150, 250, role_dict['Tackle_O'][players])
        for i in range(role_dict['Tackle_O'][players]):
            self.players.append(Player(180, player_y[i], len(self.players), 'Tackle_O'))

        # Create defenders
        for i in range(role_dict['CB'][players]):
            if i == 0:
                self.players.append(Player(225, 325, len(self.players), 'CB'))
            if i == 1:
                self.players.append(Player(225, 75, len(self.players), 'CB'))

        player_y = np.linspace(150, 250, role_dict['Tackle_D'][players])
        for i in range(role_dict['Tackle_D'][players]):
            self.players.append(Player(220, player_y[i], len(self.players), 'Tackle_D'))

        for i in range(role_dict['Safety'][players]):
            if i == 0:
                if role_dict['Safety'][players] == 1:
                    self.players.append(Player(300, 200, len(self.players), 'Safety'))
                else:
                    self.players.append(Player(300, 225, len(self.players), 'Safety'))
            if i == 1:
                self.players.append(Player(300, 175, len(self.players), 'Safety'))

        # Create the ball
        self.ball = Ball(-1, -1)

        # The stamina for the ball holder
        self.bodytouch_streak = 0
        
        # Output file name prefix
        self.prefix = prefix
        
        # Set up video writer
        os.makedirs(f"results/{self.prefix}/{self.id}/frames", exist_ok=True)
        self.vw = cv2.VideoWriter(f"results/{self.prefix}/{self.id}/demo.mp4",
                                  cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                                  10,
                                  (int(1.2 * Gameyard.w), int(1.3 * Gameyard.h)),
                                  True)

    def display(self, t):
        '''
            t: int, current time step
            -----------------------------------------
            Visualization for time step t.
        '''

        # Background figure for visualization
        self.bg_color = [0x00, 0xee, 0x00]
        self.defline_color = [0x00, 0xdd, 0xdd]
        self.start_line_color = [0x50, 0x00, 0x00]
        self.plot_bg = np.array(self.bg_color).reshape(1, 1, -1)
        self.plot_bg = self.plot_bg.repeat(Gameyard.h, 0).repeat(Gameyard.w, 1)

        # The touchdown line
        self.plot_bg[:, Gameyard.defensive_line_x - 4:Gameyard.defensive_line_x + 5, :] = self.defline_color

        # The scrimmage
        self.plot_bg[:, Gameyard.start_line_x - 4:Gameyard.start_line_x + 5, :] = self.start_line_color

        # The canvas to put players
        canvas = np.ones((int(1.3 * Gameyard.h), int(1.2 * Gameyard.w), 3), dtype=np.int32) * 200
        canvas[int(0.15 * Gameyard.h):int(1.15 * Gameyard.h) + 1, int(0.1 * Gameyard.w):int(1.1 * Gameyard.w)] = self.plot_bg
        
        # Plot players
        for player in self.players:
            cv2.circle(img=canvas, 
                       center=(int(player.x + 0.1 * Gameyard.w), int(player.y + 0.15 * Gameyard.h)), 
                       radius=Gameyard.w // 100, 
                       color=player.color, 
                       thickness=Gameyard.w // 60)
            if player.trajectory is not None:
                # TODO: optional, plot trajectory
                pass

        # Plot ball if it's being passed
        if self.ball.status == 'midair':
            cv2.circle(img=canvas,
                       center=(int(self.ball.x + 0.1 * Gameyard.w), int(self.ball.y + 0.15 * Gameyard.h)),
                       radius=Gameyard.w // 150,
                       color=self.ball.color,
                       thickness=Gameyard.w // 120)

        # Write image        
        os.makedirs(f'results/{self.prefix}/{self.id}', exist_ok=True)
        cv2.imwrite(f'results/{self.prefix}/{self.id}/frames/{self.prefix}_{self.id}_{t}_{len(self.players) // 2}v{len(self.players) // 2}.jpg', canvas)
        img = cv2.imread(f'results/{self.prefix}/{self.id}/frames/{self.prefix}_{self.id}_{t}_{len(self.players) // 2}v{len(self.players) // 2}.jpg')
        self.vw.write(img)    

    def judge_end(self, hard_end=None, cause=''):
        '''
            hard_end: str, the winner of the game if it must be ended
            cause: str, the reason the game must be ended.
            -------------------------------------------------------------
            Judge if the game should end
        '''

        # Test for hard ends
        if hard_end is not None:
            return (True, hard_end, cause)

        # Condition for offensive win: 
        # the offensive player holding the ball is beyond the defensive line
        for player in self.players:
            if player.isoffender and player.holding:
                if player.x > Gameyard.defensive_line_x:
                    return (True, 'offender', 'touchdown')
        
        # Condition for defensive win: 
        # 1. defensive player keeps bodytouch with the ball holder for long enough
        # 2. ball holder runs oout of bound
        # 3. ball pass/receiving fails
        tol = 20
        bodytouch = False
        if self.ball.status == 'held':
            for player in self.players:
                if not player.isoffender:
                    if np.sqrt((player.x - self.ball.x) ** 2 + (player.y - self.ball.y) ** 2) <= tol:
                        bodytouch = True
                        self.bodytouch_streak += 1
                        if self.bodytouch_streak >= 20:
                            return (True, 'defender', 'holder tackled')
            if not bodytouch:
                self.bodytouch_streak = 0
        
        if self.ball.y < 0 or self.ball.y > Gameyard.h:
            return (True, 'defender', 'holder out')
        
        # No winner yet: game continues
        return (False, '', '')
