import numpy as np
from nmf_methods.nmf_son.base import nmf_son as nmf_son_base
from nmf_methods.nmf_son.new import nmf_son_constrained_H
from nmf_methods.nmf_son.new import nmf_son_new


def create_toy_ex(n):
    W = np.random.rand(2, 3)
    H = np.ones((3, n))
    thres = 0.88
    id = np.argwhere(np.sum(H >= thres, axis=0))
    while id.any():
        id = np.argwhere(np.sum(H >= thres, axis=0))
        H[:, id.flatten()] = np.random.dirichlet((0.33, 0.33, 0.33), len(id)).T

    M = W @ H
    return M, W, H


# M, W, H = create_toy_ex(30)
M = np.load('../experimental/datasets/urban_small.npz')['X']
m, n = M.shape

r = 6
lam = 2
itermax = 10

EARLY_STOP = True
VERBOSE = True
SCALE_REG = False

ini_W = np.random.rand(m, r)
ini_H = np.random.rand(r, n)

# old ADMM
# W_base, H_base, fscores_base, gscores_base, lvals_base = nmf_son_base(M, ini_W.copy(), ini_H.copy(), _lambda=lam, itermax=itermax, early_stop=EARLY_STOP, verbose=VERBOSE)
#
# # old ADMM with constrained H
# W_conH, H_conH, fscores_conH, gscores_conH, lvals_conH = nmf_son_constrained_H(M, ini_W.copy(), ini_H.copy(), _lambda=lam, itermax=itermax, early_stop=EARLY_STOP, verbose=VERBOSE)

# new
W_new, H_new, fscores_new, gscores_new, lvals_new = nmf_son_new(M, ini_W.copy(), ini_H.copy(), _lambda=lam, itermax=itermax, early_stop=EARLY_STOP, verbose=VERBOSE)