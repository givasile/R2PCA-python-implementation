import timeit
import numpy as np
import scipy.linalg as linalg


def compute_U(M, r):
    # it can search for 1.5min max
    time_deadline = 90

    # assertions
    assert type(M) == np.ndarray
    assert len(M.shape) == 2
    assert type(r) == int
    D, N = M.shape
    assert r < D

    # init A with a dummy vector
    A = np.expand_dims(np.zeros([D], dtype=np.float), -1)

    deadline_reached = False
    first_time = True
    start_time = timeit.default_timer()
    while (np.linalg.matrix_rank(A) < D-r) and (not deadline_reached):
        ii = np.sort(np.random.choice(range(D), r + 1, replace=False))
        jj = np.sort(np.random.choice(range(N), r + 1, replace=False))

        if np.linalg.matrix_rank(M[np.ix_(ii, jj)]) == r:
            ai = linalg.null_space(M[np.ix_(ii, jj)].T)
            tmp = np.zeros((D, 1))
            tmp[np.ix_(ii, [0])] = ai
            A = np.concatenate([A, tmp], axis=1)

            # if first add vector, overwrite the dummy one
            if first_time:
                first_time = False
                A = A[:, 1:]

        if timeit.default_timer() - start_time > time_deadline:
            deadline_reached = True
            print("Deadline reached")
            return None

    Uhat = linalg.null_space(A.T)
    return Uhat

def compute_theta(M, U):
    D, N = M.shape
    r = U.shape[1]
    Coeffs = np.zeros((r, N))
    for j in range(N):

        resp = 0
        start = timeit.default_timer()
        while resp == 0: # and timeit.default_timer() - start < 120 / N:
            oi = np.random.choice(D, r + 1, replace=False)
            Uoi = U[np.ix_(oi, np.arange(r))]
            xoi = M[np.ix_(oi, [j])]

            A = np.dot(Uoi.T, Uoi)
            if np.linalg.matrix_rank(A) < A.shape[0]:
                Coeffs[:, j] = np.NaN
            else:
                Coeffs[:, j] = np.squeeze(np.linalg.solve(A, np.dot(Uoi.T, xoi) ))

            xoiPerp = xoi - np.expand_dims(np.dot(Uoi, Coeffs[:, j]), -1)


            if np.linalg.norm(xoiPerp) / np.linalg.norm(xoi) < 1e-9:
                resp = 1
    return Coeffs


def RPCA(M, r):
    U_hat = compute_U(M, r)
    if U_hat is not None:
        theta_hat = compute_theta(M, U_hat)
        L_hat = np.dot(U_hat, theta_hat)
    else:
        L_hat = None
    return L_hat