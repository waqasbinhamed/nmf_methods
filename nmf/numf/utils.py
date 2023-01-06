import numpy as np


def create_Up(m, p):
    """Creates unimodal restriction matrix."""
    D = np.diag(np.ones(p + 1)) + np.diag(-1 * np.ones(p), -1)
    if p < m - 1:
        Dt = np.diag(np.ones(m - p - 1)) + np.diag(-1 * np.ones(m - p - 2), 1)
        Up = np.block([[D, np.zeros((p + 1, m - p - 1))],
                       [np.zeros((m - p - 1, p + 1)), Dt]])
    else:
        Up = D
    return Up


def create_D(m):
    """Creates a (m - 1, m) size first order difference matrix."""
    D = np.zeros((m - 1, m))
    i, j = np.indices(D.shape)
    D[i == j] = -1
    D[i == j - 1] = 1
    return D