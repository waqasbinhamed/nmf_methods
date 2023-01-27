from scipy.signal import find_peaks


def get_neighbors(locs, m, nrad=2):
    """Returns a list containing integers close to integers in locs list."""
    vals = set()
    for i in locs:
        vals = vals.union(range(i - nrad, i + nrad + 1, 2))
    return list(vals.intersection(range(0, m)))


def get_peaks(M, nrad=2):
    """Returns a list containing all integer values in the neighborhoods of likely peaks."""
    (m, n) = M.shape
    all_peaks = list()
    for j in range(n):
        # TODO: find best parameters
        peaks, _ = find_peaks(x=M[:, j].reshape(m, ), prominence=1, width=6)
        all_peaks.extend(peaks)
    return get_neighbors(all_peaks, m, nrad=nrad)