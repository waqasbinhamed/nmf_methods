import numpy as np
from nmf_methods.nmf_son.base import update_wj, update_hj
from nmf_methods.nmf_son.utils import non_neg, calculate_gscore, nmf_son_ini, nmf_son_post_it

INNER_TOL = 1e-6


def sep_update_H(M, W, H):
    """Calculates the updated H without altering the W matrix."""
    Mj = M - W @ H
    for j in range(W.shape[1]):
        wj = W[:, j: j + 1]
        hj = H[j: j + 1, :]

        Mj = Mj + wj @ hj
        H[j: j + 1, :] = hj = update_hj(Mj, wj)
        Mj = Mj - wj @ hj

    return H


def sep_update_W(M, W, H, scaled_lambda):
    """Calculates the updated W without altering the H matrix."""
    m, rank = W.shape
    Mj = M - W @ H
    for j in range(rank):
        wj = W[:, j: j + 1]
        hj = H[j: j + 1, :]

        Mj = Mj + wj @ hj
        W[:, j: j + 1] = wj = update_wj(W, Mj, wj, hj, j, scaled_lambda)
        Mj = Mj - wj @ hj

    return W


def her_accelerated(M, W, H, lam=0.0, itermin=100, itermax=1000, early_stop=True, verbose=False, scale_reg=False):
    """Calculates NMF decomposition of the M matrix with new acceleration."""
    beta, _beta, gr, _gr, decay = 0.5, 1, 1.05, 1.01, 1.5

    fscores, gscores, lambda_vals = nmf_son_ini(M, W, H, lam, itermax, scale_reg)

    W_hat, H_hat = W, H
    for it in range(1, itermax + 1):
        # update H
        H_new = sep_update_H(M, W_hat, H)
        H_hat = non_neg(H_new + beta * (H_new - H))

        # update W
        W_new = sep_update_W(M, W, H_hat, lambda_vals[it - 1])
        W_hat = non_neg(W_new + beta * (W_new - W))

        fscores, gscores, lambda_vals, stop_now = nmf_son_post_it(M, W, H, it, fscores, gscores, lambda_vals,
                                                                  early_stop, verbose, scale_reg, lam, itermin)

        total_score = fscores[it] + lambda_vals[it - 1] * gscores[it]

        if total_score > fscores[it] + lambda_vals[it - 1] * gscores[it]:
            W_hat, H_hat = W_new, H_new

            _beta = beta
            beta = beta / decay
        else:
            W_new, H_new = W_hat, H_hat

            _beta = min(1, _beta * _gr)
            beta = min(_beta, beta * gr)
        W, H = W_new, H_new

        if stop_now:
            break

    return W, H, fscores[:it + 1], gscores[:it + 1], np.r_[np.NaN, lambda_vals[1: it + 1]]