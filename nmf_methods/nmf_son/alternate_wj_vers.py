import numpy as np
from nmf_methods.nmf_son.utils import non_neg, nmf_son_ini, nmf_son_post_it
from nmf_methods.nmf_son.base import update_hj

INNER_TOL = 1e-6


def func(c, diff_wis_norm, hj_norm_sq, MhT, lam):
    return 0.5 * hj_norm_sq * (np.linalg.norm(c) ** 2) - np.dot(MhT.T, c) + lam * np.sum(diff_wis_norm)


def line_search(c, diff_wis, hj_norm_sq, MhT, lam, grad, beta=0.75, itermax=1000):
    t = 1
    k = 0
    val = func(c, diff_wis, hj_norm_sq, MhT, lam)
    while k < itermax and func(c - t * grad, diff_wis, hj_norm_sq, MhT, lam) > val:
        t *= beta
        k += 1
    return t


def update_wj_subgrad(W, Mj, z, hj, j, lam, itermax=1000):
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
        t = line_search(z, diff_wis_norm, hj_norm_sq, MhT, lam, grad)
        new_z = non_neg(z - t * grad)

        if np.linalg.norm(z - new_z) / np.linalg.norm(z) < INNER_TOL:
            break
    return new_z


def update_wj_smoothing(W, Mj, new_z, hj, j, lam, mu=1, itermax=1000):
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
        t = line_search(z, z - wi_arr, hj_norm_sq, MhT, lam, grad)

        new_z = non_neg(z - t * grad)

        if np.linalg.norm(z - new_z) / np.linalg.norm(z) < INNER_TOL:
            break
    return new_z


def alt_wj_solver(M, W, H, lam=0.0, itermin=100, itermax=1000, wj_ver='subgrad', early_stop=True, verbose=False, scale_reg=False):
    """Calculates NMF decomposition of the M matrix with andersen acceleration options."""
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
            if wj_ver == 'subgrad':
                W[:, j: j + 1] = wj = update_wj_subgrad(W, Mj, wj, hj, j, lambda_vals[it - 1])
            elif wj_ver == 'nesterov':
                W[:, j: j + 1] = wj = update_wj_smoothing(W, Mj, wj, hj, j, lambda_vals[it - 1])
            else:
                raise ValueError("Please specify which methods you want to use to calculate w_j.")

            Mj = Mj - wj @ hj

        fscores, gscores, lambda_vals, stop_now = nmf_son_post_it(M, W, H, it, fscores, gscores, lambda_vals,
                                                                  early_stop, verbose, scale_reg, lam, itermin)
        if stop_now:
            break

    return W, H, fscores[:it + 1], gscores[:it + 1], np.r_[np.NaN, lambda_vals[1: it + 1]]
