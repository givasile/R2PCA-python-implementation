import timeit
import numpy as np
import generate_data as gd
import R2PCA

# main part
D = 200
N = 200
r = 5
p = 0.01  # probability of non-zero entry in S
mu = 10  # coherence of basis matrix U
assert 1 <= mu <= D/r
v = 10  # variance of sparse matrix

s = np.ceil(p*N)  # nof corrupted entries per row in S

# generate data
L, U, theta = gd.compute_L(D, r, mu)
S = gd.compute_S(D, N, r, s, v)
M = L + S

# run R2PCA
start = timeit.default_timer()
L_hat = R2PCA.RPCA(M, r)
end = timeit.default_timer()

# error
if L_hat is None:
    print("Unable to find low dimension")
else:
    print("Error: %.2f, Time: %.2f" %(np.linalg.norm(L - L_hat) / np.linalg.norm(L), end-start))
