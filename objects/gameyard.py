import numpy as np
import cv2
import random
from objects.agents import Ball, Player
from matplotlib import pyplot

class Gameyard:
    w = 500
    h = 400
    defensive_line_x = 400
    start_line_x = 200
    def __init__(self, players=1):
        role_dict = {
            'QB':       [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'WR':       [-1, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            'Tackle_O': [-1, 0, 0, 1, 2, 3, 3, 4, 5, 6, 7, 8],
            'Safety':   [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
            'CB':       [-1, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            'Tackle_D': [-1, 0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7]
        }

        # offensive init
        self.players = [Player(125, 200, 0, 'QB')]
        for i in range(role_dict['WR'][players]):
            if i == 0:
                self.players.append(Player(175, 325, len(self.players), 'WR'))
            if i == 1:
                self.players.append(Player(175, 75, len(self.players), 'WR'))
        player_y = np.linspace(150, 250, role_dict['Tackle_O'][players])
        for i in range(role_dict['Tackle_O'][players]):
            self.players.append(Player(180, player_y[i], len(self.players), 'Tackle_O'))

        # defensive init
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

        # # players: number of players on each side
        # if players == 1:
        #     self.players = [Player(100, 200, 0, 'QB'),
        #                     Player(300, 200, 1, 'Safety')]
        # else:
        #     self.players = [Player(100, 200, 0, 'QB')]
        #     player_y = np.linspace(100, 300, players - 1)
        #     self.players += [Player(150, player_y[i], i, random.choice(['Tackle_O', 'WR'])) for i in range(players - 1)]
        #     self.players += [Player(250, player_y[i], players + i, 'Tackle_D') for i in range(players - 1)] 
        #     self.players.append(Player(300, 200, 0, 'Safety'))
        self.ball = Ball(-1, -1)
        
    def display(self):
        # Background figure for visualization
        self.bg_color = [0x00, 0xee, 0x00]
        self.defline_color = [0x00, 0xdd, 0xdd]
        self.start_line_color = [0x50, 0x00, 0x00]
        self.plot_bg = np.array(self.bg_color).reshape(1, 1, -1)
        self.plot_bg = self.plot_bg.repeat(Gameyard.h, 0).repeat(Gameyard.w, 1)
        cv2.line(img=self.plot_bg,
                 pt1=(Gameyard.defensive_line_x, 0), 
                 pt2=(Gameyard.defensive_line_x, Gameyard.h - 1), 
                 color=self.defline_color, 
                 thickness=Gameyard.w // 75)

        cv2.line(img=self.plot_bg,
                 pt1=(Gameyard.start_line_x, 0), 
                 pt2=(Gameyard.start_line_x, Gameyard.h - 1), 
                 color=self.start_line_color, 
                 thickness=Gameyard.w // 75)

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

        # TODO: formal figure display
        # cv2.imshow('2v2', self.plot_bg)
        
        cv2.imwrite(f'{len(self.players) // 2}v{len(self.players) // 2}.jpg', canvas)
    
    def judge_end(self):
        # Judge if the game should end

        # Condition for offensive win: the offensive player holding the ball 
        # is beyond the defensive line
        for player in self.players:
            if player.isoffender and player.holding:
                if player.x > Gameyard.defensive_line_x:
                    return (True, 'offensive')
        
        # Condition for defensive win: at least one defensive player 
        # is close enough to the ball 
        tol = 5
        for player in self.players:
            if not player.isoffender:
                if np.sqrt((player.x - self.ball.x) ** 2 + (player.y - self.ball.y) ** 2) <= tol:
                    return (True, 'defensive')
        
        if self.ball.y < 0 or self.ball.y > Gameyard.h:
            return (True, 'defensive')
        # No winner yet: game continues
        return (False, '')


if __name__ == '__main__':
    game_id = 0

    # Sample game 1
    game1 = Gameyard(11)
    game1.display()

    # Sample game 2
    game2 = Gameyard(10)
    game2.display()

    # Sample game 3
    game3 = Gameyard(9)
    game3.display()
