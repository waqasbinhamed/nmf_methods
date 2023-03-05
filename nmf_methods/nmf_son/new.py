import numpy as np
from nmf_methods.nmf_son.base import update_wj
from nmf_methods.nmf_son.utils import non_neg, calculate_gscore, nmf_son_ini, nmf_son_post_it

INNER_TOL = 1e-6
EPS = 1e-12


def sep_update_H(M, W, H):
    """Calculates the updated H without altering the W matrix."""

    def proj_simplex(y):
        return non_neg(y - np.max((np.cumsum(-1 * np.sort(-y, axis=0), axis=0) - 1) /
                                  np.arange(1, y.shape[0] + 1).reshape(y.shape[0], 1), axis=0))

    WtW = W.T @ W
    H = proj_simplex(H - (WtW @ H - W.T @ M) / np.linalg.norm(WtW, ord=2))

    return H


def sep_update_W(M, W, H, scaledlam):
    """Calculates the updated W without altering the H matrix."""
    m, rank = W.shape
    Mj = M - W @ H
    for j in range(rank):
        wj = W[:, j: j + 1]
        hj = H[j: j + 1, :]

        Mj = Mj + wj @ hj
        W[:, j: j + 1] = wj = update_wj(W, Mj, wj, hj, j, scaledlam)
        Mj = Mj - wj @ hj

    return W


def prox(a, c, v):
    vc = v - c
    out = v - vc / np.max([1, np.linalg.norm(vc / a)])
    return out


def sep_update_W_new(M, W, H, scaledlam):
    """Calculates the updated W without altering the H matrix."""
    m, rank = W.shape
    Mj = M - W @ H

    for j in range(rank):
        wj = W[:, j: j + 1]
        hj = H[j: j + 1, :]
        hj_norm_sq = hj @ hj.T
        Mj = Mj + wj @ hj

        w_bar = (Mj @ hj.T) / (hj_norm_sq + EPS)
        prox_w_sum = 0
        for k in range(rank):
            if k != j:
                prox_w_sum += prox(scaledlam / (hj_norm_sq * rank + EPS), W[:, k: k + 1], w_bar)

        prox_in = non_neg(w_bar)
        W[:, j: j + 1] = wj = (prox_w_sum + prox_in) / rank
        Mj = Mj - wj @ hj
    return W


def constrained_H(M, W, H, lam=0.0, itermin=100, itermax=1000, early_stop=True, verbose=False, scale_reg=False):
    """Calculates NMF decomposition of the M matrix with new acceleration."""

    fscores, gscores, lambda_vals = nmf_son_ini(M, W, H, lam, itermax, scale_reg)

    for it in range(1, itermax + 1):
        # update H
        H = sep_update_H(M, W, H)

        # update W
        W = sep_update_W(M, W, H, lambda_vals[it - 1])

        fscores, gscores, lambda_vals, stop_now = nmf_son_post_it(M, W, H, it, fscores, gscores, lambda_vals,
                                                                  early_stop, verbose, scale_reg, lam, itermin)
        if stop_now:
            break

    return W, H, fscores[:it + 1], gscores[:it + 1], np.r_[np.NaN, lambda_vals[1: it + 1]]


def new(M, W, H, lam=0.0, itermin=100, itermax=1000, early_stop=True, verbose=False, scale_reg=False):
    """Calculates NMF decomposition of the M matrix with new acceleration."""

    fscores, gscores, lambda_vals = nmf_son_ini(M, W, H, lam, itermax, scale_reg)

    for it in range(1, itermax + 1):
        # update H
        H = sep_update_H(M, W, H)

        # update W
        W = sep_update_W_new(M, W, H, lambda_vals[it - 1])

        fscores, gscores, lambda_vals, stop_now = nmf_son_post_it(M, W, H, it, fscores, gscores, lambda_vals,
                                                                  early_stop, verbose, scale_reg, lam, itermin)
        if stop_now:
            break

    return W, H, fscores[:it + 1], gscores[:it + 1], np.r_[np.NaN, lambda_vals[1: it + 1]]