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
            score = dist - aggressive_coef * max_horizontal_dist

        if role_a == 'CB' and role_b == 'WR':
            # Za: offensive     Zb: defensive
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            
            score = dist

        # WR/CB-TD/TO pair
        if role_a == 'WR' and role_b == 'Tackle_D':
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
            score = dist - 0.1 * aggressive_coef * max_horizontal_dist

        if role_a == 'CB' and role_b == 'Tackle_O':
            # Za: offensive     Zb: defensive
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            score = -dist

        # Tackle strategy
        if role_a == 'Tackle_D':
            # Za: offensive     Zb: defensive
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            score = dist

        if role_a == 'Tackle_O':
            # Za: offensive     Zb: defensive
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            score = -dist

        # QB strategy
        if role_a == 'QB':
            H = Za.shape[1]
            dist = np.inf
            for h in range(H):
                dist_h = np.linalg.norm(Za[0:2, h] - Zb[0:2, h])
                if dist_h < dist:
                    dist = dist_h
            score = dist

        # TODO: strategy for safety
        
        return score



def ChooseAction(probs):
    N = probs.size
    r = np.random.rand()
    cumulative = np.cumsum(probs)
    for a in range(N):
        if r <= cumulative[a]:
            return a

