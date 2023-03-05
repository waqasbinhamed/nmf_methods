import numpy as np
from nmf_methods.nmf_son.utils import non_neg, nmf_son_ini, nmf_son_post_it
from nmf_methods.nmf_son.base import update_hj
# np.seterr(all='raise')

INNER_TOL = 1e-6


def find_anderson_coeff(R_history):
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


def anderson_acc_1d(first_it_flag, anderson_win, val, new_val, prev_vals=None, residuals=None):
    if first_it_flag:
        prev_vals = new_val.copy()
        residuals = new_val - val
    else:
        prev_vals = np.c_[prev_vals, new_val]
        residuals = np.c_[residuals, new_val - val]
        if residuals.shape[1] > anderson_win:
            prev_vals = prev_vals[:, 1:]
            residuals = residuals[:, 1:]

        coeffs = find_anderson_coeff(residuals)
        new_val = (prev_vals @ coeffs).reshape(val.shape[0], 1)
    return new_val, prev_vals, residuals


def anderson_acc_2d(first_it_flag, anderson_win, val_arr, new_val_arr, prev_val_arr=None, residual_arr=None):
    m, n = val_arr.shape

    if first_it_flag:
        prev_val_arr = new_val_arr.copy()
        residual_arr = new_val_arr - val_arr
    else:
        prev_val_arr = np.dstack((prev_val_arr, new_val_arr))
        residual_arr = np.dstack((residual_arr, new_val_arr - val_arr))
        if residual_arr.shape[2] > anderson_win:
            prev_val_arr = prev_val_arr[:, :, 1:]
            residual_arr = residual_arr[:, :, 1:]
        for i in range(n):
            coeffs = find_anderson_coeff(residual_arr[:, i, :])
            new_val_arr[:, i: i + 1] = (prev_val_arr[:, i, :] @ coeffs).reshape(m, 1)

    return new_val_arr, prev_val_arr, residual_arr


def update_wj_anderson_z(W, Mj, new_z, hj, j, lam, itermax=1000, anderson_win=2):
    """Calculates the w_j vector with anderson acceleration of only z."""
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
        tmp_arr = zeta_arr / lam - ci_arr

        tmp_norm = np.linalg.norm(tmp_arr, axis=0)
        norm_mask = tmp_norm > 1
        new_wi_arr[:, norm_mask] = zeta_arr[:, norm_mask] - lam * (tmp_arr[:, norm_mask] / tmp_norm[norm_mask])
        new_wi_arr[:, ~norm_mask] = zeta_arr[:, ~norm_mask] - lam * tmp_arr[:, ~norm_mask]

        new_z = (rho * (new_wf + new_w0) + rho * np.sum(new_wi_arr, axis=1, keepdims=True) + yf + y0
                 + np.sum(yi_arr, axis=1, keepdims=True)) / (rho * (2 + num_edges))

        if np.linalg.norm(new_z - z) / np.linalg.norm(z) < INNER_TOL:
            break

        # anderson acceleration on z variable
        if it > 0:
            new_z, prev_z, res_z = anderson_acc_1d(False, anderson_win, z, new_z, prev_z, res_z)
        else:
            new_z, prev_z, res_z = anderson_acc_1d(True, anderson_win, z, new_z)

        new_yf = yf + rho * (new_wf - new_z)
        new_y0 = y0 + rho * (new_w0 - new_z)
        new_yi_arr = yi_arr + rho * (new_wi_arr - new_z)

        # # anderson acceleration on yf, y0, yi_arr variable
        # if it > 0:
        #     new_yf, prev_yf, res_yf = anderson_acc_1d(False, anderson_win, yf, new_yf, prev_yf, res_yf)
        #     new_y0, prev_y0, res_y0 = anderson_acc_1d(False, anderson_win, y0, new_y0, prev_y0, res_y0)
        #     new_yi_arr, prev_yi_arr, res_yi_arr = anderson_acc_2d(False, anderson_win, yi_arr, new_yi_arr, prev_yi_arr,
        #                                                           res_yi_arr)
        # else:
        #     new_yf, prev_yf, res_yf = anderson_acc_1d(True, anderson_win, yf, new_yf)
        #     new_y0, prev_y0, res_y0 = anderson_acc_1d(True, anderson_win, y0, new_y0)
        #     new_yi_arr, prev_yi_arr, res_yi_arr = anderson_acc_2d(True, anderson_win, yi_arr, new_yi_arr)

    return new_z


def update_wj_anderson_all(W, Mj, new_z, hj, j, lam, itermax=1000, anderson_win=2):
    """Calculates the w_j vector with anderson acceleration of w0, wf, wi's, z."""
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
        tmp_arr = zeta_arr / lam - ci_arr

        tmp_norm = np.linalg.norm(tmp_arr, axis=0)
        norm_mask = tmp_norm > 1
        new_wi_arr[:, norm_mask] = zeta_arr[:, norm_mask] - lam * (tmp_arr[:, norm_mask] / tmp_norm[norm_mask])
        new_wi_arr[:, ~norm_mask] = zeta_arr[:, ~norm_mask] - lam * tmp_arr[:, ~norm_mask]

        # anderson acceleration on wf, w0, wi_arr variable
        if it > 0:
            new_wf, prev_wf, res_wf = anderson_acc_1d(False, anderson_win, z, new_wf, prev_wf, res_wf)
            new_w0, prev_w0, res_w0 = anderson_acc_1d(False, anderson_win, z, new_w0, prev_w0, res_w0)
            new_wi_arr, prev_wi_arr, res_wi_arr = anderson_acc_2d(False, anderson_win, z, new_wi_arr, prev_wi_arr,
                                                                  res_wi_arr)
        else:
            new_wf, prev_wf, res_wf = anderson_acc_1d(True, anderson_win, z, new_wf)
            new_w0, prev_w0, res_w0 = anderson_acc_1d(True, anderson_win, z, new_w0)
            new_wi_arr, prev_wi_arr, res_wi_arr = anderson_acc_2d(True, anderson_win, z, new_wi_arr)

        new_z = (rho * (new_wf + new_w0) + rho * np.sum(new_wi_arr, axis=1, keepdims=True) + yf + y0
                 + np.sum(yi_arr, axis=1, keepdims=True)) / (rho * (2 + num_edges))

        if np.linalg.norm(new_z - z) / np.linalg.norm(z) < INNER_TOL:
            break

        # anderson acceleration on z variable
        if it > 0:
            new_z, prev_z, res_z = anderson_acc_1d(False, anderson_win, z, new_z, prev_z, res_z)
        else:
            new_z, prev_z, res_z = anderson_acc_1d(True, anderson_win, z, new_z)

        new_yf = yf + rho * (new_wf - new_z)
        new_y0 = y0 + rho * (new_w0 - new_z)
        new_yi_arr = yi_arr + rho * (new_wi_arr - new_z)

        # # anderson acceleration on yf, y0, yi_arr variable
        # if it > 0:
        #     new_yf, prev_yf, res_yf = anderson_acc_1d(False, anderson_win, yf, new_yf, prev_yf, res_yf)
        #     new_y0, prev_y0, res_y0 = anderson_acc_1d(False, anderson_win, y0, new_y0, prev_y0, res_y0)
        #     new_yi_arr, prev_yi_arr, res_yi_arr = anderson_acc_2d(False, anderson_win, yi_arr, new_yi_arr,
        #                                                           prev_yi_arr, res_yi_arr)
        # else:
        #     new_yf, prev_yf, res_yf = anderson_acc_1d(True, anderson_win, yf, new_yf)
        #     new_y0, prev_y0, res_y0 = anderson_acc_1d(True, anderson_win, y0, new_y0)
        #     new_yi_arr, prev_yi_arr, res_yi_arr = anderson_acc_2d(True, anderson_win, yi_arr, new_yi_arr)

    return new_z


def anderson_accelerated(M, W, H, lam=0.0, itermin=100, itermax=1000, anderson_ver='z', anderson_win=2, early_stop=True, verbose=False, scale_reg=False):
    """Calculates NMF decomposition of the M matrix with anderson acceleration options."""
    r = W.shape[1]

    fscores, gscores, lambda_vals = nmf_son_ini(M, W, H, lam, itermax, scale_reg)

    Mj = M - W @ H
    for it in range(1, itermax + 1):
        for j in range(r):
            wj = W[:, j: j + 1]
            hj = H[j: j + 1, :]

            Mj = Mj + wj @ hj

            # update h_j
            H[j: j + 1, :] = hj = update_hj(Mj, wj)

            # update w_j
            if anderson_ver == 'z':
                W[:, j: j + 1] = wj = update_wj_anderson_z(W, Mj, wj, hj, j, lambda_vals[it - 1],
                                                           anderson_win=anderson_win)
            elif anderson_ver == 'all':
                W[:, j: j + 1] = wj = update_wj_anderson_all(W, Mj, wj, hj, j, lambda_vals[it - 1],
                                                             anderson_win=anderson_win)
            else:
                raise ValueError("Please specify which version of the anderson accelerated approach you want to use.")

            Mj = Mj - wj @ hj

        fscores, gscores, lambda_vals, stop_now = nmf_son_post_it(M, W, H, it, fscores, gscores, lambda_vals,
                                                                  early_stop, verbose, scale_reg, lam, itermin)
        if stop_now:
            break

    return W, H, fscores[:it + 1], gscores[:it + 1], np.r_[np.NaN, lambda_vals[1: it + 1]]
