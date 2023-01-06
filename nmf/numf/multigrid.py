import math
import numpy as np


def create_R(m):
    n = int(math.floor(m / 2))
    R = np.zeros((n, m))
    for i in range(n):
        R[i, 2 * i: 2 * i + 3] = 1/3
    return R


def get_fine_p(ps, scaling_factor=2):
    return np.array(ps) * scaling_factor
