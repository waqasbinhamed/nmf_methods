import numpy as np
from nmf.numf.utils import create_D, create_Up
np.random.seed(42)


def numf(M, W, H, pvals=None, l2=0, beta=0, iters=100, save_file=None, verbose=True):
    """Runs the NuMF algorithm to factorize the M vector into unimodal peaks."""
    (m, n) = M.shape
    r = W.shape[1]  # rank

    for it in range(1, iters + 1):
        pouts = numf_it(H, M, W, l2, m, n, pvals, r, beta)
        if it % 5 == 0 or it == iters:
            if verbose:
                print(f"Loss: {np.linalg.norm(M - W @ H, 'fro') / np.linalg.norm(M, 'fro')}")
            if save_file is not None:
                with open(save_file, 'wb') as fout:
                    np.savez_compressed(fout, W=W, H=H, pouts=pouts)
                if verbose:
                    print(f'W and H matrices saved in {save_file} in {it} iterations.')
    return W, H, pouts


def numf_it(H, M, W, l2, m, n, pvals, r, beta):
    pouts = list()
    Mi = M - W @ H
    for i in range(r):
        wi = W[:, i].reshape(m, 1)
        hi = H[i, :].reshape(1, n)

        Mi = Mi + wi @ hi

        # updating hi
        H[i, :] = update_hi(Mi, wi, n)

        # updating wi
        W[:, i], pout = update_wi(Mi, wi, hi, m, pvals, l2, beta)
        pouts.append(pout)

        Mi = Mi - wi @ hi
    return pouts


def update_wi(Mi, wi, hi, m, pvals=None, l2=0, beta=0):
    """Updates the value of w(i) column as part of BCD."""
    wmin = np.empty((m, 1))
    min_score = np.Inf
    min_p = 0

    hi_norm = np.linalg.norm(hi) ** 2
    Mhi = Mi @ hi.T

    if pvals is None:
        pvals = range(1, m, 2)  # trying all p values

    for p in pvals:
        # creating Up matrix
        Up = create_Up(m, p)
        invUp = np.linalg.inv(Up)
        Q = hi_norm * (invUp.T @ invUp)
        if l2 != 0:
            D = create_D(m)
            tmp = D @ invUp
            tmp2 = tmp.T @ tmp
            Q = Q + l2 * (np.linalg.norm(Q, 'fro') / np.linalg.norm(tmp2, 'fro')) * tmp2
        if beta != 0:
            tmp3 = invUp.T @ invUp
            Q = Q + beta * (np.linalg.norm(Q, 'fro') / np.linalg.norm(tmp3, 'fro')) * tmp3
        _p = invUp.T @ Mhi
        b = invUp.T @ np.ones((m, 1))

        # accelerated projected gradient
        ynew = apg(Up @ wi, Q, _p, b)

        score = 0.5 * np.dot((Q @ ynew).T, ynew) - np.dot(_p.T, ynew)
        if score < min_score:
            min_p = p
            min_score = score
            wmin = invUp @ ynew
    return wmin.reshape(m, ), min_p


def apg(y, Q, _p, b, itermax=100):
    """Runs acceraled projected gradient."""
    k = 1
    yhat = ynew = y
    norm_Q = np.linalg.norm(Q, ord=2)
    while (np.linalg.norm(ynew - y) > 1e-3 or k == 1) and k < itermax:  # temporary
        y = ynew
        z = yhat - (Q @ yhat - _p) / (norm_Q + 1e-16)
        nu = calculate_nu(b, z)
        # TODO: try alternate optimization methods
        ynew = z - nu * b
        ynew[ynew < 0] = 0
        yhat = ynew + ((k - 1) / (k + 2)) * (ynew - y)
        k += 1
    return ynew


def calculate_nu(b, z):
    nz_idx = b >= 0
    nzb = b[nz_idx]
    nzz = z[nz_idx]

    idx = np.argsort(-z / b, 0)
    return np.max((np.cumsum(nzz[idx] * nzb[idx]) - 1) / np.cumsum(nzb[idx] * nzb[idx]))


def update_hi(Mi, wi, n):
    """Updates the value of h(i) row as part of BCD."""
    tmp = Mi.T @ wi
    tmp[tmp < 0] = 0
    hi = tmp / (np.linalg.norm(wi) ** 2)
    return hi.reshape(1, n)

