import numpy as np
import scipy.linalg as linalg


def compute_S(D, N, r, s, v):
    s = int(s)
    S = np.zeros((D, N))
    for i in range(D):
        S[i, np.random.choice(N, s, replace=False)] = np.random.normal(s, 1)

    for j in range(N):
        idx = np.where(S[:, j] != 0)[0]
        non_zeros = len(idx)
        if non_zeros > D - r - 1:
            new_zeros = non_zeros - (D-r-1)
            idx1 = idx[np.random.choice(non_zeros, new_zeros, replace=False)]
            S[np.ix_(idx1, [j])] = 0

    return S


def compute_L(D, r, mu):

    # generate basis with coherence in [mu-0.5, mu+0.5]
    U = basis_with_coherence(D, r, mu)
    linalg.orth(U)
    mu = compute_coherence(U)
    print("Coherence of U is: %.2f" % (mu))

    theta = np.random.randn(r, D) # coefficients of low rank component
    L = np.dot(U, theta)  # Low rank component
    return L, U, theta


def basis_with_coherence(D, r, mu):
    """Dummy method for finding basis U with coherence: mu-.5 <= c <= mu+0.5
Keep generating U with c >= mu + 0.5 until we are lucky to find one with mu-.5 <= c <= mu+0.5.
nn
    :param D: 
    :param r: 
    :param mu: 
    :returns: 
    :rtype: 

    """
    
    U = basis_with_coherence_bigger_than_mu(D, r, mu - 0.5)
    while compute_coherence(U) > mu + 0.5:
        U = basis_with_coherence_bigger_than_mu(D, r, mu - 0.5)
        
    return U
    

def basis_with_coherence_bigger_than_mu(D, r, mu):
    """Generate a basis matrix with coherence c >= mu.

    :param D: 
    :param r: 
    :param mu: 
    :returns: 
    :rtype: 

    """
    U = np.random.randn(D, r)
    c = compute_coherence(U)
    i = 1
    it = 1
    while c < mu:
        U[i-1, :] = U[i-1, :] / 10**it # divide row i by 10^it
        i += 1
        c = compute_coherence(U)
        if i == D+1:
            i = 1
            it = it + 1
    return U


def compute_coherence(U):
    P = np.dot(np.dot(U, np.linalg.inv(np.dot(U.T, U))), U.T) # P: DxD
    D, r = U.shape

    projections = np.zeros((D, 1))
    for i in range(D):
        # create i-th canonical vector
        ei = np.zeros((D, 1))
        ei[i] = 1

        # projection to the i-th canonical vector
        # projections[i] = np.linalg.norm(np.dot(P, ei), 2)**2
        projections[i] = np.linalg.norm(P[:, i], 2)**2

    mu = D/r * np.max(projections)
    return mu
