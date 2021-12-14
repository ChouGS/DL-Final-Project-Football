import numpy as np

def LCP(M, q, max_iters=50):
    # Function for computing solutions to the LCP (M,q)
    # Returns a tuple (z,w,ret), where
    # ret = -1 if illspecified problem
    #   (w,z arbitrary in this case)
    # ret = 0 if algorithm terminates in ray termination
    #   (w,z arbitrary in this case)
    # ret = 1 if algorithm terminates successfully with a solution:
    #   w = Mz + q
    #   w >= 0
    #   z >= 0
    #   w.dot(z) = 0
    # This function is an implementation of Lemke's principle pivoting algorithm, as described in Section 2.2 of
    # "Linear Complementarity, Linear and Nonlinear Programming", by Katta G. Murty
    #
    # Assume that the vector q is nondegenerate

    n = M.shape[0]
    if n != M.shape[1] or n != q.shape[0]:
        # Dimension mismatch in input data, return failure
        z = np.zeros((n, 1))
        w = M.dot(z) + q
        ret = -1
        return (z, w, ret)
    else:
        # Check if trivial solution exists
        if np.all(q > -1e-6):
            z = np.zeros((n, 1))
            w = q
            ret = 1
            return(z, w, ret)

        # Assign variable indexing:
        # index var
        #  0     w0
        # ...    ...
        # n-1    w_{n-1}
        #  n     z0
        # ...	 ...
        # 2n-1	 z_{n-1}
        # 2n	 z0

        T = np.hstack((np.eye(n), -M, -np.ones((n, 1)), q))
        basis = np.arange(n)

        # Initialize starting point
        t = np.argmin(q)
        basis[t] = 2 * n

        # This block of code assigns entering var necessary value so that exiting var is 0
        pivot = T[t, :] / T[t, 2 * n]
        T -= np.outer(T[:, 2*n], pivot)
        T[t, :] = pivot
        #
        entering_ind = t+n

        iters = 0
        ret = 1  # assume for now that we won't ray terminate

        while 2*n in basis and iters < max_iters:
            d = T[:, entering_ind]
            wrong_dir = d <= 0
            ratios = T[:, -1] / d
            ratios[wrong_dir] = np.inf
            t = np.argmin(ratios)
            if ~ np.all(wrong_dir):
                # This block of code assigns entering var necessary value so that exiting var is 0
                pivot = T[t, :] / T[t, entering_ind]
                T -= np.outer(T[:, entering_ind], pivot)
                T[t, :] = pivot
                #
                exiting_ind = basis[t]
                basis[t] = entering_ind
                if exiting_ind >= n:
                    entering_ind = exiting_ind - n
                else:
                    entering_ind = exiting_ind + n
            else:
                ret = 0
                break
        vars = np.zeros((2*n+1, 1))
        vars[basis, 0] = T[:, -1]
        w = vars[0:n]
        z = vars[n:2 * n]

        return (z, w, ret)

def LCP_lemke_howson(A, B, max_iters=50):
    # Function for computing solutions to the Bimatrix game defined by cost matrices A > 0, B > 0
    # Returns a tuple (x,y), where x and y are the mixed equilibrium strategies for player 1 and player 2, respectively.

    # This function is an implementation of the Lemke-Howson method, as described in Section 2.5 of
    # "Linear Complementarity, Linear and Nonlinear Programming", by Katta G. Murty
    n = A.shape[0]
    m = A.shape[1]

    # Ensure A and B are strictly positive
    if (np.any(A <= 0)):
        A += abs(np.min(A)) + 1.0  # any positive constant would do
    if (np.any(B <= 0)):
        B += abs(np.min(B)) + 1.0  # any positive constant would do
    
    # TODO
    o = B.shape[0]
    p = B.shape[1]
    T = np.hstack(( np.vstack(( np.eye(m), np.zeros((o,m)) )), np.vstack(( np.zeros((m,n)), np.eye(n) )),
                    np.vstack(( np.zeros((n,o)), -B.T)), np.vstack(( -A, np.zeros((m,m)) )), -np.ones(( n+p,1 )) ))
    
    row = T.shape[0]
    col = T[:, row]
    tp = -np.inf
    for i, j in enumerate(col):
        if j != 0 and j > tp:
            tp = j
            q_ind = i
    basis = np.arange(row)
    T[q_ind,:] = T[q_ind,:] / T[q_ind,row]
    for i in range(row):
        if i != q_ind:
            T[i,:] = T[i,:] - T[q_ind,:] * T[i,row]
    basis[q_ind] = row
    
    row_2 = q_ind + row
    tp_2 = -np.inf
    col_2 = T[:,row_2]
    for i,j in enumerate(col_2):
        if j != 0 and j > tp_2:
            tp_2 = j
            q_ind_2 = i    
    T[q_ind_2,:] = T[q_ind_2,:] / T[q_ind_2,row_2]
    for i in range(row):
        if i!=q_ind_2:
            T[i,:] = T[i,:] - T[q_ind_2,:] * T[i,row_2]    
    basis[q_ind_2] = row_2
    
    entering_ind = row + q_ind_2
    iters = 0
    ret = 1
    if q_ind_2!=row or q_ind_2!=0:
        while iters < max_iters and row in basis and 0 in basis:
            d = T[:, entering_ind]
            wrong_dir = d <= 0
            ratios = T[:, -1] / d
            ratios[wrong_dir] = np.inf
            t = np.argmin(ratios)
            if ~ np.all(wrong_dir):
                pivot = T[t, :] / T[t, entering_ind]
                T -= np.outer(T[:, entering_ind], pivot)
                T[t, :] = pivot
                exiting_ind = basis[t]
                basis[t] = entering_ind
                if exiting_ind >= row:
                    entering_ind = exiting_ind - row
                else:
                    entering_ind = exiting_ind + row
            else:
                ret = 0
                break
    vars = np.zeros((row*2, 1))
    vars[basis, 0] = T[:,-1]
    x = vars[row:row+n] / sum(vars[row:row+n])[0]
    y = vars[row+n:] / sum(vars[row+n:])[0]
    
    return (x,y)

