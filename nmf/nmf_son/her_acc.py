import numpy as np
from nmf_son.base import update_wj
from nmf_son.utils import non_neg, calculate_gscore

ES_TOL = 1e-5
INNER_TOL = 1e-6


def sep_update_H(M, W, H):
    """Calculates the updated H without altering the W matrix."""
    Mj = M - W @ H
    for j in range(W.shape[1]):
        wj = W[:, j: j + 1]
        hj = H[j: j + 1, :]

        Mj = Mj + wj @ hj
        H[j: j + 1, :] = hj = non_neg(wj.T @ Mj) / (np.linalg.norm(wj) ** 2)
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


def nmf_son_acc(M, W, H, _lambda=0.0, itermax=1000, early_stop=False, verbose=False):
    """Calculates NMF decomposition of the M matrix with new acceleration."""
    beta, _beta, gr, _gr, decay = 0.5, 1, 1.05, 1.01, 1.5

    m, n = M.shape
    rank = W.shape[1]

    fscores = np.full((itermax + 1,), np.NaN)
    gscores = np.full((itermax + 1,), np.NaN)
    lambda_vals = np.full((itermax + 1,), np.NaN)

    fscores[0] = np.linalg.norm(M - W @ H, 'fro')
    gscores[0] = calculate_gscore(W)

    scaled_lambda = lambda_vals[0] = (fscores[0] / gscores[0]) * _lambda

    best_score = np.Inf
    W_best = np.zeros((m, rank))
    H_best = np.zeros((rank, n))

    W_hat, H_hat = W, H
    for it in range(1, itermax + 1):
        # update H
        H_new = sep_update_H(M, W_hat, H)
        H_hat = non_neg(H_new + beta * (H_new - H))

        # update W
        W_new = sep_update_W(M, W, H_hat, scaled_lambda)
        W_hat = non_neg(W_new + beta * (W_new - W))

        fscores[it] = 0.5 * np.linalg.norm(M - W @ H, 'fro') ** 2
        gscores[it] = calculate_gscore(W)
        total_score = fscores[it] + scaled_lambda * gscores[it]

        if total_score > fscores[it] + lambda_vals[it - 1] * gscores[it]:
            W_hat, H_hat = W_new, H_new

            _beta = beta
            beta = beta / decay
        else:
            W_new, H_new = W_hat, H_hat

            _beta = min(1, _beta * _gr)
            beta = min(_beta, beta * gr)
        W, H = W_new, H_new

        if total_score > best_score:
            best_score = total_score
            W_best = W
            H_best = H

        if early_stop and it > 2:
            old_score = fscores[it - 1] + lambda_vals[it - 2] * gscores[it - 1]
            if abs(old_score - total_score) / old_score < ES_TOL:
                break

        scaled_lambda = lambda_vals[it] = (fscores[it] / gscores[it]) * _lambda

        if verbose:
            print(f'Iteration: {it}, f={fscores[it]}, g={gscores[it]},  total={total_score}')

    return W_best, H_best, W, H, fscores[:it + 1], gscores[:it + 1], np.r_[np.NaN, lambda_vals[1: it + 1]]