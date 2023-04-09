import os
import time
import numpy as np
from nmf_methods.nmf_son.new import new as nmf_son_new
from nmf_methods.nmf_son.utils import save_results
import csv

np.random.seed(42)
np.set_printoptions(precision=3)

EARLY_STOP = True
VERBOSE = False
SCALE_REG = True

max_iters = 10000

results_csv_fp = '../../experimental/saved_models/multi_size/output2.csv'
fieldnames = ['n', 'rank', 'lambda', 'time_taken', 'fscore', 'gscore', 'scaled_lambda']
with open(results_csv_fp, mode='w') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

jasper_full = np.load('../../experimental/datasets/jasper_full.npz')['X']
jasper_3d = jasper_full.reshape(-1, 100, 100, order='F')

ini_filepath = '../../experimental/saved_models/multi_size/r{}_ini.npz'
save_filepath = '../../experimental/saved_models/multi_size/r{}_l{}_mit{}.npz'
lambda_vals = [10, 100]

m, _ = jasper_full.shape
dims = [3, 5, 7, 10, 12, 15, 20]
for dim in dims:
    M_3d = jasper_3d[:, :dim, :dim]
    M = M_3d.reshape(m, -1, order='F')
    m, n = M.shape
    r = n
    if os.path.exists(ini_filepath.format(r)):
        data = np.load(ini_filepath.format(r))
        ini_W = data['ini_W']
        ini_H = data['ini_H']
    else:
        ini_W = np.random.rand(m, r)
        ini_H = np.random.rand(r, n)
        with open(ini_filepath.format(r), 'wb') as fout:
            np.savez_compressed(fout, ini_W=ini_W, ini_H=ini_H)

    for _lam in lambda_vals:
        start_time = time.time()
        W, H, fscores, gscores, lvals = nmf_son_new(M, ini_W.copy(), ini_H.copy(), lam=_lam, itermax=max_iters,
                                                    early_stop=EARLY_STOP, verbose=VERBOSE, scale_reg=SCALE_REG)
        time_taken = time.time() - start_time
        save_results(save_filepath.format(r, _lam, max_iters), W, H, fscores, gscores, lvals)
        with open(results_csv_fp, 'a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow({'n': n,
                             'rank': r,
                             'lambda': _lam,
                             'time_taken': time_taken,
                             'fscore': fscores[-1],
                             'gscore': gscores[-1],
                             'scaled_lambda': lvals[-2]})
            print(r, _lam)
