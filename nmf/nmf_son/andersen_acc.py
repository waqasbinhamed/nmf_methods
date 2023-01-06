import numpy as np
from nmf_son.utils import non_neg, calculate_gscore
np.seterr(all='raise')


ES_TOL = 1e-5
INNER_TOL = 1e-6


def find_andersen_coeff(R_history):
    """
    Compute the combination coefficients alpha_i in Anderson acceleration, i.e., solve
        argmin sum_{i=0}^m alpha_i r_{k-i},  s.t. sum_{i=0}^{m} alpha_i = 1
    Solve using the equivalent least square problem by eliminating the constraint
    """
    nc = R_history.shape[1]

    # Construct least square matrix
    if nc == 1:
        c = np.ones(1)
    else:
        Y = R_history[:, 1:] - R_history[:, 0:-1]
        b = R_history[:, -1]
        q, r = np.linalg.qr(Y)

        z = np.linalg.solve(r, q.T @ b)
        c = np.r_[z[0], z[1:] - z[0:-1], 1 - z[-1]]

    return c


def andersen_acc_1d(first_it_flag, andersen_win, val, new_val, prev_vals=None, residuals=None):
    if first_it_flag:
        prev_vals = new_val.copy()
        residuals = new_val - val
    else:
        prev_vals = np.c_[prev_vals, new_val]
        residuals = np.c_[residuals, new_val - val]
        if residuals.shape[1] > andersen_win:
            prev_vals = prev_vals[:, 1:]
            residuals = residuals[:, 1:]

        coeffs = find_andersen_coeff(residuals)
        new_val = (prev_vals @ coeffs).reshape(val.shape[0], 1)
    return new_val, prev_vals, residuals


def andersen_acc_2d(first_it_flag, andersen_win, val_arr, new_val_arr, prev_val_arr=None, residual_arr=None):
    m, n = val_arr.shape

    if first_it_flag:
        prev_val_arr = new_val_arr.copy()
        residual_arr = new_val_arr - val_arr
    else:
        prev_val_arr = np.dstack((prev_val_arr, new_val_arr))
        residual_arr = np.dstack((residual_arr, new_val_arr - val_arr))
        if residual_arr.shape[2] > andersen_win:
            prev_val_arr = prev_val_arr[:, :, 1:]
            residual_arr = residual_arr[:, :, 1:]
        for i in range(n):
            coeffs = find_andersen_coeff(residual_arr[:, i, :])
            new_val_arr[:, i: i + 1] = (prev_val_arr[:, i, :] @ coeffs).reshape(m, 1)

    return new_val_arr, prev_val_arr, residual_arr


def update_wj_andersen_z(W, Mj, new_z, hj, j, _lambda, itermax=1000, andersen_win=2):
    """Calculates the w_j vector with andersen acceleration of only z."""
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

        # andersen acceleration on z variable
        if it > 0:
            new_z, prev_z, res_z = andersen_acc_1d(False, andersen_win, z, new_z, prev_z, res_z)
        else:
            new_z, prev_z, res_z = andersen_acc_1d(True, andersen_win, z, new_z)

        new_yf = yf + rho * (new_wf - new_z)
        new_y0 = y0 + rho * (new_w0 - new_z)
        new_yi_arr = yi_arr + rho * (new_wi_arr - new_z)

        # # andersen acceleration on yf, y0, yi_arr variable
        # if it > 0:
        #     new_yf, prev_yf, res_yf = andersen_acc_1d(False, andersen_win, yf, new_yf, prev_yf, res_yf)
        #     new_y0, prev_y0, res_y0 = andersen_acc_1d(False, andersen_win, y0, new_y0, prev_y0, res_y0)
        #     new_yi_arr, prev_yi_arr, res_yi_arr = andersen_acc_2d(False, andersen_win, yi_arr, new_yi_arr, prev_yi_arr, res_yi_arr)
        # else:
        #     new_yf, prev_yf, res_yf = andersen_acc_1d(True, andersen_win, yf, new_yf)
        #     new_y0, prev_y0, res_y0 = andersen_acc_1d(True, andersen_win, y0, new_y0)
        #     new_yi_arr, prev_yi_arr, res_yi_arr = andersen_acc_2d(True, andersen_win, yi_arr, new_yi_arr)

    return new_z


def update_wj_andersen_all(W, Mj, new_z, hj, j, _lambda, itermax=1000, andersen_win=2):
    """Calculates the w_j vector with andersen acceleration of w0, wf, wi's, z."""
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

        # andersen acceleration on wf, w0, wi_arr variable
        if it > 0:
            new_wf, prev_wf, res_wf = andersen_acc_1d(False, andersen_win, z, new_wf, prev_wf, res_wf)
            new_w0, prev_w0, res_w0 = andersen_acc_1d(False, andersen_win, z, new_w0, prev_w0, res_w0)
            new_wi_arr, prev_wi_arr, res_wi_arr = andersen_acc_2d(False, andersen_win, z, new_wi_arr, prev_wi_arr, res_wi_arr)
        else:
            new_wf, prev_wf, res_wf = andersen_acc_1d(True, andersen_win, z, new_wf)
            new_w0, prev_w0, res_w0 = andersen_acc_1d(True, andersen_win, z, new_w0)
            new_wi_arr, prev_wi_arr, res_wi_arr = andersen_acc_2d(True, andersen_win, z, new_wi_arr)


        new_z = (rho * (new_wf + new_w0) + rho * np.sum(new_wi_arr, axis=1, keepdims=True) + yf + y0
                 + np.sum(yi_arr, axis=1, keepdims=True)) / (rho * (2 + num_edges))

        if np.linalg.norm(new_z - z) / np.linalg.norm(z) < INNER_TOL:
            break

        # andersen acceleration on z variable
        if it > 0:
            new_z, prev_z, res_z = andersen_acc_1d(False, andersen_win, z, new_z, prev_z, res_z)
        else:
            new_z, prev_z, res_z = andersen_acc_1d(True, andersen_win, z, new_z)

        new_yf = yf + rho * (new_wf - new_z)
        new_y0 = y0 + rho * (new_w0 - new_z)
        new_yi_arr = yi_arr + rho * (new_wi_arr - new_z)

        # # andersen acceleration on yf, y0, yi_arr variable
        # if it > 0:
        #     new_yf, prev_yf, res_yf = andersen_acc_1d(False, andersen_win, yf, new_yf, prev_yf, res_yf)
        #     new_y0, prev_y0, res_y0 = andersen_acc_1d(False, andersen_win, y0, new_y0, prev_y0, res_y0)
        #     new_yi_arr, prev_yi_arr, res_yi_arr = andersen_acc_2d(False, andersen_win, yi_arr, new_yi_arr, prev_yi_arr, res_yi_arr)
        # else:
        #     new_yf, prev_yf, res_yf = andersen_acc_1d(True, andersen_win, yf, new_yf)
        #     new_y0, prev_y0, res_y0 = andersen_acc_1d(True, andersen_win, y0, new_y0)
        #     new_yi_arr, prev_yi_arr, res_yi_arr = andersen_acc_2d(True, andersen_win, yi_arr, new_yi_arr)

    return new_z


def nmf_son_z_accelerated(M, W, H, _lambda=0.0, itermax=1000, andersen_win=2, early_stop=False, verbose=False):
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
            W[:, j: j + 1] = wj = update_wj_andersen_z(W, Mj, wj, hj, j, scaled_lambda, andersen_win=andersen_win)
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


def nmf_son_all_accelerated(M, W, H, _lambda=0.0, itermax=1000, andersen_win=2, early_stop=False, verbose=False):
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
            W[:, j: j + 1] = wj = update_wj_andersen_all(W, Mj, wj, hj, j, scaled_lambda, andersen_win=andersen_win)
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
