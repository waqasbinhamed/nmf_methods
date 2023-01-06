import numpy as np
from nmf_son.utils import non_neg, calculate_gscore


ES_TOL = 1e-5
INNER_TOL = 1e-6


def update_wj(W, Mj, new_z, hj, j, _lambda, itermax=1000):
    """Calculates the w_j vector."""
    m, r = W.shape

    rho = 1
    num_edges = (r * (r - 1)) / 2
    ci_arr = np.delete(W, j, axis=1)

    new_wi_arr = np.zeros((m, r - 1))

    new_yi_arr = np.random.rand(m, r - 1)
    new_yf = np.random.rand(m, 1)
    new_y0 = np.random.rand(m, 1)

    hj_norm_sq = np.linalg.norm(hj) ** 2
    for it in range(itermax):
        z = new_z
        yf = new_yf
        y0 = new_y0
        yi_arr = new_yi_arr

        new_wf = (Mj @ hj.T - yf + rho * z) / (rho + hj_norm_sq)
        new_w0 = non_neg(z - y0 / rho)

        zeta_arr = z - yi_arr / rho
        tmp_arr = zeta_arr / _lambda - ci_arr

        tmp_norm = np.linalg.norm(tmp_arr, axis=0)
        norm_mask = tmp_norm > 1
        new_wi_arr[:, norm_mask] = zeta_arr[:, norm_mask] - _lambda * (tmp_arr[:, norm_mask] / tmp_norm[norm_mask])
        new_wi_arr[:, ~norm_mask] = zeta_arr[:, ~norm_mask] - _lambda * tmp_arr[:, ~norm_mask]

        new_z = (rho * (new_wf + new_w0) + rho * np.sum(new_wi_arr, axis=1, keepdims=True) + yf + y0
                 + np.sum(yi_arr, axis=1, keepdims=True)) / (rho * (2 + num_edges))

        if np.linalg.norm(new_z - z) / np.linalg.norm(z) < INNER_TOL:
            break

        new_yf = yf + rho * (new_wf - new_z)
        new_y0 = y0 + rho * (new_w0 - new_z)
        new_yi_arr = yi_arr + rho * (new_wi_arr - new_z)
    return new_z


def nmf_son(M, W, H, _lambda=0.0, itermax=1000, early_stop=True, verbose=False):
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
            W[:, j: j + 1] = wj = update_wj(W, Mj, wj, hj, j, scaled_lambda)

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
