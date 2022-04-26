import numpy as np

def EvaluateTrajectories(Za, Zb, player_a, player_b, mode, offender_pattern):
    '''
        Za: 4*H, [x, y, vx, vy] tuples that marks the status of player A
        Zb: 4*H, [x, y, vx, vy] tuples that marks the status of player B
        player_a: object of class Player, player a
        player_b: object of class Player, player b
        mode: choose{'1v1', 'mv1'}, current game mode
        offender_pattern: choose{'H', 'L'} the hyper-parameter OP
        -------------------------------------------------------
        Returns the score of trajectory Za against trajectory Zb in the given mode and OP.
        The final score will be based on:
        - The minimum distance between player A and player B.
        - The minimum distance between player A.
    '''

    # Set AC according to OP
    if offender_pattern == 'H':
        aggressive_coef = 100
    elif offender_pattern == 'L':
        aggressive_coef = 1
    else:
        raise ValueError("offender_pattern must be either 'H' or 'L'.")
    
    assert mode in ['1v1', 'mv1']

    # Fetch information from players
    role_a = player_a.role
    role_b = player_b.role
    start_x = player_a.x
    holding_a = player_a.holding
    holding_b = player_b.holding

    # 1v1 mode
    if mode == '1v1':
        # WR-CB pairs
        if role_a == 'WR' and role_b == 'CB':
            # Calculate minimum distance along trajectories
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            
            # Calculate maximum displacement
            max_horizontal_dist = -500
            for h in range(H):
                if Za[1, h] > 400 or Za[1, h] < 0:
                    break
                max_horizontal_dist = max(max_horizontal_dist, Za[0, h] - start_x)
            
            # Calculate cost value
            loss = - dist - 100 * aggressive_coef * max_horizontal_dist

        elif role_a == 'CB' and role_b == 'WR':
            # Calculate minimum distance along trajectories
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            
            # Calculate cost value
            loss = 0.00000001 * dist

        # WR/CB-TD/TO pair
        elif role_a == 'WR' and role_b == 'Tackle_D':
            # Calculate minimum distance along trajectories
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            
            # Calculate maximum displacement
            max_horizontal_dist = -500
            for h in range(H):
                if Za[1, h] > 400 or Za[1, h] < 0:
                    break
                max_horizontal_dist = max(max_horizontal_dist, Za[0, h] - start_x)
            
            # Calculate cost value
            loss = -dist - 10 * aggressive_coef * max_horizontal_dist

        elif role_a == 'CB' and role_b == 'Tackle_O':
            # Calculate minimum distance along trajectories
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            
            # Calculate cost value
            loss = 1e10 #-/dist

        # Tackle strategy
        elif role_a == 'Tackle_D':
            # Calculate minimum distance along trajectories
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            
            # Calculate cost value
            loss = dist if role_b == 'Tackle_O' else 100000 * dist

        elif role_a == 'Tackle_O':
            # Calculate minimum distance along trajectories
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h

            # Calculate cost value
            loss = 0.001 * dist if role_b == 'Tackle_D' else dist

        # QB strategy
        elif role_a == 'QB':
            # Calculate minimum distance along trajectories
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h

            # Calculate maximum displacement
            max_horizontal_dist = -500
            for h in range(H):
                if Za[1, h] > 400 or Za[1, h] < 0:
                    break
                max_horizontal_dist = max(max_horizontal_dist, Za[0, h] - start_x)
            
            # Calculate cost value
            loss = -dist - 10 * aggressive_coef * max_horizontal_dist

        elif role_a == 'WR' and role_b == 'Safety':
            # Calculate minimum distance along trajectories
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h

            # Calculate maximum displacement
            max_horizontal_dist = -500
            for h in range(H):
                if Za[1, h] > 400 or Za[1, h] < 0:
                    break
                max_horizontal_dist = max(max_horizontal_dist, Za[0, h] - start_x)
            
            # Calculate cost value
            loss = -dist - 200 * aggressive_coef * max_horizontal_dist
        
        elif role_a == 'CB' and role_b == 'QB':
            # Calculate minimum distance along trajectories
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            loss = 1e10 #-/dist

    # mv1 mode
    if mode == 'mv1':
        # Ball-holding WR strategy 
        if role_a == 'WR' and holding_a:
            # Calculate minimum distance along trajectories
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            
            # Calculate maximum displacement
            max_horizontal_dist = -500
            for h in range(H):
                if Za[1, h] > 400 or Za[1, h] < 0:
                    break
                max_horizontal_dist = max(max_horizontal_dist, Za[0, h] - start_x)
            
            # Calculate cost value
            loss = - dist - 100 * aggressive_coef * max_horizontal_dist
        
        # Other offensive players' strategy
        elif role_a == 'Tackle_O' or role_a == 'QB' or role_a == 'WR':
            # Calculate minimum distance along trajectories
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h

            # Calculate maximum displacement
            max_horizontal_dist = -500
            for h in range(H):
                if Za[1, h] > 400 or Za[1, h] < 0:
                    break
                max_horizontal_dist = max(max_horizontal_dist, Za[0, h] - start_x)
            
            # Calculate cost value
            loss = dist
        
        # Defensive strategy
        else:
            # Against ball holder
            if holding_b:
                # Calculate minimum distance along trajectories
                H = Za.shape[1]
                dist = np.inf
                for h in range(H):
                    dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                    if dist_h < dist:
                        dist = dist_h
                
                # Calculate cost value
                loss = 0.000001 * dist
            
            # Against other offensive players
            else:
                # Calculate minimum distance along trajectories
                H = Za.shape[1]
                dist = np.inf
                for h in range(H):
                    dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                    if dist_h < dist:
                        dist = dist_h
                
                # Calculate cost value
                loss = 1e8 / dist
    
    # Avoid all-zero columns in the final cost matrix
    if loss == 0:
        loss = 1e-6
    return loss


def EvaluateTrajectoriesForSafety(Za, Zb, p, q, traj_list, players, mode='1v1', a=10, b=50, c=1):
    '''
        Za: 4*H, [x, y, vx, vy] tuples that marks the status of player A
        Zb: 4*H, [x, y, vx, vy] tuples that marks the status of player B
        p: the id of the safety himself
        q: the id of the offender to be considered
        traj_list: list of trajectories
        players: list of Player instances, all the players on the gameyard
        mode: choose{'1v1', 'mv1'}, current game mode
        offender_pattern: choose{'H', 'L'} the hyper-parameter OP
        -------------------------------------------------------
        Returns the score of trajectory Za against trajectory Zb in the given mode and OP.
        ! This function is designed only for safeties.
        The final score will be based on:
        - The minimum distance between player q and the safety.
        - The minimum distance between player q and all other defenders.
        - The horizontal displacement of player q.
    '''

    assert mode in ['1v1', 'mv1']
    # Fetch information of players
    start_x = players[q].x
    holding_b = players[q].holding
    H = Za.shape[1]
    N = len(traj_list[0])

    # 1v1 mode
    if mode == '1v1':
        # Calculate minimum distance (between player q and other defenders) 
        # along trajectories
        dist_close = np.inf
        for k in range(len(traj_list)):
            # filter out teammates
            if players[k].isoffender:
                continue
            if k == p:
                continue
            else:
                for h in range(H):
                    for l in range(N):
                        dist_h = np.linalg.norm(traj_list[k][0][l][0:2, h] - Zb[0:2, h])
                        if dist_h < dist_close:
                            dist_close = dist_h
        
        # Calculate minimum distance (between player q and the safety)
        dist = np.inf
        for h in range(H):
            dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
            if dist_h < dist:
                dist = dist_h

        # Calculate maximum displacement
        max_horizontal_dist = -500
        for h in range(H):
            if Zb[1, h] > 400 or Zb[1, h] < 0:
                break
            max_horizontal_dist = max(max_horizontal_dist, Zb[0, h] - start_x)
        
        loss = -a * dist_close - b * max_horizontal_dist + c * dist

    # mv1 mode: Basically the same as EvaluateTrajectories
    else:
        # Against the ball holder
        if holding_b:
            H = Za.shape[1]

            # Calculate minimum distance (between player q and the safety)
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h

            # Calculate cost value
            loss = dist
        
        # Against other offenders
        else:
            H = Za.shape[1]

            # Calculate minimum distance (between player q and the safety)
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
                    
            # Calculate cost value
            loss = -dist

    # Avoid all-zero columns in the final cost matrix
    if loss == 0:
        loss = 1e-6
    return loss


def ChooseAction(probs):
    '''
        probs: a list of 0~1 real numbers, the probability the agent will choose each action
        ---------------------------------------
        Does sampling according to the provided probability vector.
    '''
    
    N = probs.size
    r = np.random.rand()
    cumulative = np.cumsum(probs)
    for a in range(N):
        if r <= cumulative[a]:
            return a


def LoseBallProb(x):
    '''
        x: real, the distance between the ball passer and the nearest opponent
        ---------------------------------------
        Returns the likelihood for the ball passing to fail.
    '''
    return 21 / (20 * (x + 1)) - 1 / 20
