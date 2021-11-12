from objects.gameyard import Gameyard
from objects.agents import Player, Ball

# Initialize game and timer
tick = 0
game = Gameyard(players=1)

# Game starts
while (True):
    # The ball is granted to an offensive player at the 15th time step
    if tick == 15:
        game.players[0].mod_holding_state(True, game.ball)

    # Players make their decisions every 5 time steps
    # Need to record whether a ball-pass occurs
    # If so, frreze the receipient 
    if tick % 5 == 0:
        receipient_id = None

        for player in game.players:
            (hold, rec_id) = player.decision(game.players)
            if not hold and rec_id is not None:
                assert receipient_id is None
                receipient_id = rec_id
        
        if receipient_id is not None:
            for player in game.players:
                if player.id == receipient_id:
                    player.freeze()

    # Players and balls move every 1 time step
    for player in game.players:
        player.motion()
    game.ball.motion()

    # Check whether the game comes to end
    end, winner = game.judge_end()

    if end:
        # TODO: to visualize game results
        print(f'Game over, {winner}s win.')
        break

    tick += 1
    
    
    
