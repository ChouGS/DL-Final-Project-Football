import numpy as np
import cv2
from agents import Ball, Player
from matplotlib import pyplot

class Gameyard:
    def __init__(self, players=1):
        # players: number of players on each side

        global game_id              # Not useful
        self.game_id = game_id      # in actual
        game_id += 1                # 

        self.w = 500
        self.h = 400
        self.defensive_line_x = 400

        # TODO: formal initial positions
        if players == 1:
            self.players = [Player(100, 200, 0, 'offender'),
                            Player(300, 200, 1, 'defender')]
        else:
            player_y = np.linspace(100, 300, players)
            self.players = [Player(100, player_y[i], i, 'offender') for i in range(players)] + \
                           [Player(300, player_y[i], players + i, 'defender') for i in range(players)] 
        self.ball = Ball(-1, -1)

        # Background figure for visualization
        self.bg_color = [0xff, 0xf2, 0xcc]
        self.defline_color = [0x00, 0xdd, 0xdd]
        self.plot_bg = np.array(self.bg_color).reshape(1, 1, -1)
        self.plot_bg = self.plot_bg.repeat(self.h, 0).repeat(self.w, 1)
        cv2.line(img=self.plot_bg,
                 pt1=(self.defensive_line_x, 0), 
                 pt2=(self.defensive_line_x, self.h - 1), 
                 color=self.defline_color, 
                 thickness=self.w // 50)
        
    def display(self):
        for player in self.players:
            cv2.circle(img=self.plot_bg, 
                       center=(int(player.x), int(player.y)), 
                       radius=self.w // 80, 
                       color=player.color, 
                       thickness=self.w // 50)
            if player.trajectory is not None:
                # TODO: optional, plot trajectory
                pass

        if self.ball.status == 'midair':
            cv2.circle(img=self.plot_bg,
                       center=(int(self.ball.x), int(self.ball.y)),
                       radius=self.w // 150,
                       color=self.ball.color,
                       thickness=self.w // 120)

        # TODO: formal figure display
        # cv2.imshow(str(self.game_id), self.plot_bg)
        cv2.imwrite(str(self.game_id) + '.jpg', self.plot_bg)
    
    def judge_end(self):
        # Judge if the game should end

        # Condition for offensive win: the offensive player holding the ball 
        # is beyond the defensive line
        for player in self.players:
            if player.role == 'offender' and player.holding:
                if player.x > self.defensive_line_x:
                    return (True, 'offensive')
        
        # Condition for defensive win: at least one defensive player 
        # is close enough to the ball 
        tol = 5
        for player in self.players:
            if player.role == 'defender':
                if np.sqrt((player.x - self.ball.x) ** 2 + (player.y - self.ball.y) ** 2) <= tol:
                    return (True, 'defensive')
        
        # No winner yet: game continues
        return (False, '')


if __name__ == '__main__':
    game_id = 0

    # Sample game 1
    game1 = Gameyard(3)
    game1.display()

    # Sample game 2
    game2 = Gameyard(1)
    game2.ball.mod_status('midair')
    game2.ball.x = 200
    game2.ball.y = 200
    game2.display()

    # Sample game 3
    game3 = Gameyard(2)
    game3.players[1].mod_holding_state(True, game3.ball)
    game3.display()
