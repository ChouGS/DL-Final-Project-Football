import numpy as np

def EvaluateTrajectories(Za, Zb, mode='1v1', aggressive_coef=1):
    assert mode in ['1v1', 'mv1']
    # import pdb
    # pdb.set_trace()
    if mode == '1v1':
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
            max_horizontal_dist = max(max_horizontal_dist, Za[0, h])

        score = dist - aggressive_coef * max_horizontal_dist

        return score


def ChooseAction(probs):
    N = probs.size
    r = np.random.rand()
    cumulative = np.cumsum(probs)
    for a in range(N):
        if r <= cumulative[a]:
            return a

