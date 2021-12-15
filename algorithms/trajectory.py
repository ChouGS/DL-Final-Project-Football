import numpy as np

from algorithms.LCPs import LCP

def GenerateTrajectory(z0, goal, H, dt, u_max, u_penalty, bound_x, bound_y, bound_v):
    '''
        z0: [x, y, vx, vy] tuple, the current status of the player
        goal: N * 2, the coordinates of the preset target points
        H: int, length of generated trajectories
        dt: real, the frequency used for simulating the dynamics
        u_max: real, boundaries for the accelerations
        u_penalty: real, acceleration penalties for too-close opponents
        bound_x: (real, real), boundaries of the x coordinates along the generated trajectory
        bound_y: (real, real), boundaries of the y coordinates along the generated trajectory
        bound_v: (real, real), boundaries of the velocity along the generated trajectory
        -----------------------------------------------------------------------------------------
        Generate a trajectory of discrete lenght H,
        for a pointmass system defined by the state z, starting at
        position z0, defined to evolve according to the dynamics
        z_{t+1} = A z_{t} + B u_{t}
    '''

    n = 4
    m = 2
    A = np.eye(4)
    A[0, 2] = dt
    A[1, 3] = dt
    B = np.zeros((n, m))
    B[0, 0] = 0.5 * dt * dt
    B[1, 1] = 0.5 * dt * dt
    B[2, 0] = dt
    B[3, 1] = dt

    # The trajectory should be chosen such as to minimize the objective
    # 0.5 * (z_H[0:2] - goal).dot(z_H[0:2] - goal) + \sum_{h=0}^{H-1} u_penalty * 0.5 * u_h.dot(u_h)

    # The position of the pointmass should stay within the box centered at (0,0) with edge_length equal to
    # 2*box_half_width. The position of the pointmass at each stage h is defined to be first two dimensions of z_h

    # The absolute value of each dimension of the control input at stage h, u_h, should be less than u_max

    # FIRST:
    # Define x := [u_0', z_1', u_1', z_2', ..., u_{H-1}', z_H']'
    # Set up a QP:

    # min_x   0.5 x' * Q * x + q'*x
    #  s.t.    A x + b = 0
    #          lb <= x < = ub (lb or ub may be +/- infinity for some dimensions)

    Q_u = u_penalty * np.eye(H * m)
    q_u = np.zeros((H*m, 1))
    Q_z = np.zeros((H * n, H * n))
    q_z = np.zeros((H*n, 1))
    Q_z[-n: - n + 2, -n: - n + 2] = np.eye(2)
    q_z[-n: - n + 2] = -goal.reshape(-1, 1)
    Q = np.block([
        [Q_u, np.zeros((H*m, H*n))],
        [np.zeros((H*n, H*m)), Q_z]
    ])
    q = np.vstack([q_u, q_z])

    # Define D, d, such that [D -I] [u z]  d = 0
    # I.E. z = Du + d
    # BB = np.vstack([np.eye(m), B])
    # AA = np.vstack([np.zeros((m, n)), A])
    d = np.zeros((H * n, 1))
    d[0:n] = A.dot(z0).reshape(-1, 1)
    for h in range(H-1):
        d[(h+1) * n:(h + 2) * n] = A.dot(d[h * n:(h+1) * n])

    D = np.zeros((H * n, H * m))
    for h in range(H):
        D[h*n:(h+1)*n, h*m:(h+1)*m] = B
        for j in range(h-1, -1, -1):
            D[h*n:(h+1)*n, j*m:(j+1)*m] = A.dot(D[h*n:(h+1)*n, (j+1)*m:(j+2)*m])

    lb_u = -u_max*np.ones((H * m, 1))
    ub_u = u_max*np.ones((H * m, 1))

    lb_z = -np.inf*np.ones((H * n, 1))
    ub_z = np.inf*np.ones((H * n, 1))

    # Constrain position of pointmass to lie within box
    for h in range(H):
        lb_z[h * n] = bound_x[0]
        lb_z[h * n + 1] = bound_y[0]
        ub_z[h * n] = bound_x[1]
        ub_z[h * n + 1] = bound_y[1]
        ub_z[h * n + 2:h * n + 4] = bound_v

    # At this point, solving the following QP results in our desired trajectory
    #                                 min_x 0.5*x'*Q*x + x'*q,
    # s.t. bound constraints          lb <= x <= ub
    # and  equality constraints       [D -I]x + d = 0

    # Now time to relate cost and constraints on z variables to cost and constraints on u variables
    # by substituting z = D u + d

    QQ_u = Q_u + D.T.dot(Q_z.dot(D))
    qq_u = q_u + D.T.dot(Q_z.dot(d)+q_z)

    # But only want to grab dimensions which have non-infinite upper or lower bounds:

    G = np.vstack([D, -D])
    g = np.vstack([d - lb_z, ub_z - d])

    inf_mask = np.squeeze(np.isinf(g))
    G = G[~inf_mask, :]
    g = g[~inf_mask, :]

    # Now reduced QP can be solved instead:
    #                                min_u 0.5*u'*QQ_u*u + u'*qq_u
    # s.t. bound constraints l       lb_u <= u <= ub_u
    #      inequality constraints    G*u + g >= 0    
    
    M = np.vstack((QQ_u, G, -np.eye(len(qq_u))))
    M_r = np.hstack((-G.T,np.eye(len(qq_u))))
    M_r = np.vstack((M_r,np.zeros((M.shape[0]-M.shape[1],M.shape[0]-M.shape[1]))))
    M = np.hstack((M,M_r))
    q = np.vstack((qq_u+0.5*QQ_u.dot(lb_u)+0.5*QQ_u.T.dot(lb_u),G.dot(lb_u)+g,ub_u-lb_u))

    v_opt = LCP(M, q)[0]
    u_opt = v_opt[0:len(qq_u)]+lb_u
    u_opt = u_opt.reshape((len(qq_u),1))

    # Once u^* is found, find the corresponding z^* and return
    z_opt = D.dot(u_opt) + d
    Z = np.zeros((4, H))
    for h in range(H):
        Z[:, h] = z_opt[h*n:(h+1)*n].reshape(-1)
    return Z
