import numpy as np

def EvaluateTrajectories(Za, Zb, role_a, role_b, start_x, mode='1v1', aggressive_coef=1):
    assert mode in ['1v1', 'mv1']
    # import pdb
    # pdb.set_trace()
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
            # Za: offensive     Zb: defensive
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            loss = -dist

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
            loss = dist

        # Tackle strategy
        elif role_a == 'Tackle_D':
            # Za: offensive     Zb: defensive
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            loss = -dist

        elif role_a == 'Tackle_O':
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

            loss = dist # - 0.1 * aggressive_coef * max_horizontal_dist

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
            loss = -dist - aggressive_coef * max_horizontal_dist

        # TODO: strategy for safety

        elif role_a == 'WR' and role_b == 'Safety':
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            loss = -0.01 * dist
        
        elif role_a == 'CB' and role_b == 'QB':
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            loss = -0.01 * dist

        return loss
    
def EvaluateTrajectoriesForSafety(Za, Zb, p, q, i, j, traj_list, players, start_x, mode='1v1', a=10, b=5, c=10):
    assert mode in ['1v1', 'mv1']
    if mode == '1v1':
        H = Za.shape[1]
        N = len(traj_list[0])
        if N > 2:
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
            loss = a * dist_close + b * max_horizontal_dist - c * dist

        else:
            loss = - EvaluateTrajectories(Zb, Za, 'QB', 'Tackle_D', start_x, mode='1v1')
    return loss


def ChooseAction(probs):
    N = probs.size
    r = np.random.rand()
    cumulative = np.cumsum(probs)
    for a in range(N):
        if r <= cumulative[a]:
            return a
