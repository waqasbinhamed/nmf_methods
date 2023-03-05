import numpy as np

EARLY_STOP_TOL = 1e-6


def non_neg(arr):
    """Returns non-negative projection of array."""
    arr[arr < 0] = 0
    return arr


def calculate_fscore(M, W, H):
    return 0.5 * np.linalg.norm(M - W @ H, 'fro') ** 2


def calculate_gscore(W):
    """Calculates the sum of norm of the W matrix."""
    rank = W.shape[1]
    gscore = 0
    for i in range(rank - 1):
        gscore += np.sum(np.linalg.norm(W[:, i: i + 1] - W[:, i + 1:], axis=0))
    return gscore


def load_results(filepath):
    data = np.load(filepath)
    return data['W'], data['H'], data['fscores'], data['gscores'], data['lambda_vals']


def save_results(filepath, W, H, fscores, gscores, lambda_vals):
    with open(filepath, 'wb') as fout:
        np.savez_compressed(fout, W=W, H=H, fscores=fscores, gscores=gscores, lambda_vals=lambda_vals)


def normalized_similarity(W_ins):
    r = W_ins.shape[1]
    res = np.ones(shape=(r, r)) * -1
    for i in range(r):
        for j in range(r):
            res[i, j] = np.linalg.norm(W_ins[:, i] - W_ins[:, j])
        res[i, :] = res[i, :] / sum(res[i, :])
    return res


def nmf_son_ini(M, W, H, lam, itermax, scale_reg):
    fscores = np.full((itermax + 1,), np.NaN)
    gscores = np.full((itermax + 1,), np.NaN)
    lambda_vals = np.full((itermax + 1,), np.NaN)

    fscores[0] = calculate_fscore(M, W, H)
    gscores[0] = calculate_gscore(W)

    if scale_reg:
        lambda_vals[0] = (fscores[0] / gscores[0]) * lam
    else:
        lambda_vals[:] = lam

    return fscores, gscores, lambda_vals


def nmf_son_post_it(M, W, H, it, fscores, gscores, lambda_vals, early_stop, verbose, scale_reg, lam, itermin):
    stop_now = False

    fscores[it] = calculate_fscore(M, W, H)
    gscores[it] = calculate_gscore(W)
    total_score = fscores[it] + lam * gscores[it]

    if early_stop and it > itermin:
        old_score = fscores[it - 1] + lambda_vals[it - 2] * gscores[it - 1]
        if abs(old_score - total_score) / old_score < EARLY_STOP_TOL:
            stop_now = True

    if scale_reg:
        lambda_vals[it] = (fscores[it] / gscores[it]) * lam

    if verbose:
        print(f'Iteration: {it}, f={fscores[it]}, g={gscores[it]},  total={total_score}')

    return fscores, gscores, lambda_vals, stop_now