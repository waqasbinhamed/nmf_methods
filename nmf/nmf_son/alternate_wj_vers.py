import numpy as np
from nmf_son.utils import non_neg, calculate_gscore

ES_TOL = 1e-5
INNER_TOL = 1e-6


def func(c, diff_wis_norm, hj_norm_sq, MhT, _lambda):
    return 0.5 * hj_norm_sq * (np.linalg.norm(c) ** 2) - np.dot(MhT.T, c) + _lambda * np.sum(diff_wis_norm)


def line_search(c, diff_wis, hj_norm_sq, MhT, _lambda, grad, beta=0.75, itermax=1000):
    t = 1
    k = 0
    val = func(c, diff_wis, hj_norm_sq, MhT, _lambda)
    while k < itermax and func(c - t * grad, diff_wis, hj_norm_sq, MhT, _lambda) > val:
        t *= beta
        k += 1
    return t


def update_wj_subgrad(W, Mj, z, hj, j, _lambda, itermax=1000):
    def unit_norm_vec(sz):
        tau = np.zeros((sz, 1))
        tau[0] = 1
        return tau

    m = W.shape[0]
    new_z = z
    hj_norm_sq = np.linalg.norm(hj) ** 2
    MhT = Mj @ hj.T
    for k in range(itermax):
        z = new_z

        diff_wis = np.delete(W, j, axis=1) - z
        diff_wis_norm = np.linalg.norm(diff_wis, axis=0)

        norm_mask = diff_wis_norm != 0
        tmp = diff_wis.copy()
        tmp[:, norm_mask] = diff_wis[:, norm_mask] / diff_wis_norm[norm_mask]
        tmp[:, ~norm_mask] = unit_norm_vec(m)
        grad = hj_norm_sq * z - MhT + np.sum(tmp, axis=1, keepdims=True)

        # simple line search
        t = line_search(z, diff_wis_norm, hj_norm_sq, MhT, _lambda, grad)
        new_z = non_neg(z - t * grad)

        if np.linalg.norm(z - new_z) / np.linalg.norm(z) < INNER_TOL:
            break
    return new_z


def update_wj_smoothing(W, Mj, new_z, hj, j, _lambda, mu=1, itermax=1000):
    m = W.shape[0]

    wi_arr = np.delete(W, j, axis=1)

    hj_norm_sq = np.linalg.norm(hj) ** 2
    MhT = Mj @ hj.T

    for k in range(itermax):
        z = new_z

        tmp_arr = (z - wi_arr) / mu
        tmp_norm = np.linalg.norm(tmp_arr, axis=0)
        norm_mask = tmp_norm > 1
        tmp_arr[:, norm_mask] = tmp_arr[:, norm_mask] / tmp_norm[norm_mask]

        grad = hj_norm_sq * z - MhT + np.sum(tmp_arr, axis=1).reshape(m, 1)
        t = line_search(z, z - wi_arr, hj_norm_sq, MhT, _lambda, grad)

        new_z = non_neg(z - t * grad)

        if np.linalg.norm(z - new_z) / np.linalg.norm(z) < INNER_TOL:
            break
    return new_z


def nmf_son_alt(update_w_func, M, W, H, _lambda=0.0, itermax=1000, early_stop=True, verbose=False):
    """Calculates NMF decomposition of the M matrix with andersen acceleration options."""
    m, n = M.shape
    r = W.shape[1]

    fscores = np.full((itermax + 1,), np.NaN)
    gscores = np.full((itermax + 1,), np.NaN)
    lambda_vals = np.full((itermax + 1,), np.NaN)

    fscores[0] = np.linalg.norm(M - W @ H, 'fro')
    gscores[0] = calculate_gscore(W)

    scaled_lambda = lambda_vals[0] = (fscores[0] / gscores[0]) * _lambda

    best_score = np.Inf
    W_best = np.zeros((m, r))
    H_best = np.zeros((r, n))

    Mj = M - W @ H
    for it in range(1, itermax + 1):
        for j in range(r):
            wj = W[:, j: j + 1]
            hj = H[j: j + 1, :]

            Mj = Mj + wj @ hj

            # update h_j
            H[j: j + 1, :] = hj = non_neg(wj.T @ Mj) / (np.linalg.norm(wj) ** 2)

            # update w_j
            W[:, j: j + 1] = wj = update_w_func(W, Mj, wj, hj, j, scaled_lambda)

            Mj = Mj - wj @ hj

        fscores[it] = 0.5 * np.linalg.norm(M - W @ H, 'fro') ** 2
        gscores[it] = calculate_gscore(W)
        total_score = fscores[it] + scaled_lambda * gscores[it]

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
