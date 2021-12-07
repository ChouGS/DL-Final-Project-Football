import numpy as np

def EvaluateTrajectories(Za, Zb, player_a, player_b, mode, offender_pattern):
    if offender_pattern == 'H':
        aggressive_coef = 0.001
    elif offender_pattern == 'L':
        aggressive_coef = 1
    else:
        raise ValueError("offender_pattern must be either 'H' or 'L'.")
    
    assert mode in ['1v1', 'mv1']
    role_a = player_a.role
    role_b = player_b.role
    start_x = player_a.x
    holding_a = player_a.holding
    holding_b = player_b.holding

    if mode == '1v1':
        # WR-CB pairs
        if role_a == 'WR' and role_b == 'CB':
            # Za: offensive     Zb: defensive
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            
            max_horizontal_dist = -500
            for h in range(H):
                if Za[1, h] > 400 or Za[1, h] < 0:
                    break
                max_horizontal_dist = max(max_horizontal_dist, Za[0, h] - start_x)
            loss = - dist - 100 * aggressive_coef * max_horizontal_dist

        elif role_a == 'CB' and role_b == 'WR':
            # Za: defensive     Zb: offensive 
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            loss = 0.00000001 * dist

        # WR/CB-TD/TO pair
        elif role_a == 'WR' and role_b == 'Tackle_D':
            # Za: offensive     Zb: defensive
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            
            max_horizontal_dist = -500
            for h in range(H):
                if Za[1, h] > 400 or Za[1, h] < 0:
                    break
                max_horizontal_dist = max(max_horizontal_dist, Za[0, h] - start_x)
            loss = -dist - 10 * aggressive_coef * max_horizontal_dist

        elif role_a == 'CB' and role_b == 'Tackle_O':
            # Za: offensive     Zb: defensive
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            loss = 1e10 #-/dist

        # Tackle strategy
        elif role_a == 'Tackle_D':
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            loss = dist if role_b == 'Tackle_O' else 100000 * dist

        elif role_a == 'Tackle_O':
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h

            loss = 0.001 * dist if role_b == 'Tackle_D' else dist

        # QB strategy
        elif role_a == 'QB':
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            max_horizontal_dist = -500
            for h in range(H):
                if Za[1, h] > 400 or Za[1, h] < 0:
                    break
                max_horizontal_dist = max(max_horizontal_dist, Za[0, h] - start_x)
            loss = -dist - 10 * aggressive_coef * max_horizontal_dist

        elif role_a == 'WR' and role_b == 'Safety':
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            max_horizontal_dist = -500
            for h in range(H):
                if Za[1, h] > 400 or Za[1, h] < 0:
                    break
                max_horizontal_dist = max(max_horizontal_dist, Za[0, h] - start_x)
            loss = -dist - 200 * aggressive_coef * max_horizontal_dist
        
        elif role_a == 'CB' and role_b == 'QB':
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            loss = 1e10 #-/dist

    if mode == 'mv1':
        if role_a == 'WR' and holding_a:
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            
            max_horizontal_dist = -500
            for h in range(H):
                if Za[1, h] > 400 or Za[1, h] < 0:
                    break
                max_horizontal_dist = max(max_horizontal_dist, Za[0, h] - start_x)
            loss = - dist - 100 * aggressive_coef * max_horizontal_dist
        
        elif role_a == 'Tackle_O' or role_a == 'QB' or role_a == 'WR':
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h

            max_horizontal_dist = -500
            for h in range(H):
                if Za[1, h] > 400 or Za[1, h] < 0:
                    break
                max_horizontal_dist = max(max_horizontal_dist, Za[0, h] - start_x)
            loss = dist
        
        else:
            if holding_b:
                H = Za.shape[1]
                dist = np.inf
                for h in range(H):
                    dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                    if dist_h < dist:
                        dist = dist_h
                loss = 0.000001 * dist
            else:
                H = Za.shape[1]
                dist = np.inf
                for h in range(H):
                    dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                    if dist_h < dist:
                        dist = dist_h
                loss = 1e8 / dist

    return loss
    
def EvaluateTrajectoriesForSafety(Za, Zb, p, q, traj_list, players, mode='1v1', a=10, b=50, c=1):
    assert mode in ['1v1', 'mv1']
    start_x = players[q].x
    holding_b = players[q].holding
    H = Za.shape[1]
    N = len(traj_list[0])

    if mode == '1v1':
        dist_close = np.inf
        for k in range(len(traj_list)):
            if players[k].isoffender:
                continue
            if k == p:
                continue
            else:
                for h in range(H):
                    for l in range(N):
                        dist_h = np.linalg.norm(traj_list[k][l][0:2, h] - Zb[0:2, h])
                        if dist_h < dist_close:
                            dist_close = dist_h
        dist = np.inf
        for h in range(H):
            dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
            if dist_h < dist:
                dist = dist_h
        max_horizontal_dist = -500
        for h in range(H):
            if Zb[1, h] > 400 or Zb[1, h] < 0:
                break
            max_horizontal_dist = max(max_horizontal_dist, Zb[0, h] - start_x)
        loss = -a * dist_close - b * max_horizontal_dist + c * dist

    else:
        if holding_b:
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            loss = dist
        else:
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            loss = -dist

    return loss


def ChooseAction(probs):
    N = probs.size
    r = np.random.rand()
    cumulative = np.cumsum(probs)
    for a in range(N):
        if r <= cumulative[a]:
            return a

def LoseBallProb(x):
    return 21 / (20 * (x + 1)) - 1 / 20