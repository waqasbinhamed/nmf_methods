import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

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


def plot_scores(fscores, gscores, lambda_vals, plot_title=None, filename=None):
    fscores = fscores[1:]
    gscores = gscores[1:]
    lambda_vals = lambda_vals[1:]
    total_score = fscores + lambda_vals * gscores
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    if plot_title:
        fig.suptitle(plot_title, fontsize=25)

    axs[0].set_yscale('log')
    axs[0].plot(total_score, color='black', linewidth=3, label='$F(W, H)$')
    axs[0].plot(fscores, color='cyan', linewidth=1.5, label='$f(W, H)$')
    axs[0].plot(gscores, color='yellow', linewidth=1.5, label='$g(W)$')
    axs[0].set_xlabel('Iterations')
    axs[0].legend()

    fscores -= min(fscores)
    gscores -= min(gscores)
    total_score -= min(total_score)

    axs[1].set_yscale('log')
    axs[1].plot(total_score, color='black', linewidth=3, label='$F(W, H) - min(F(W, H))$')
    axs[1].plot(fscores, color='cyan', linewidth=1.5, label='$f(W, H) - min(f(W, H))$')
    axs[1].plot(gscores, color='yellow', linewidth=1.5, label='$g(W) - min(g(W))$')
    axs[1].set_xlabel('Iterations')
    axs[1].legend()

    if filename:
        fig.savefig(filename)
        plt.close()


def plot_separate_H(H, img_size, figsize, fontsize, num_rows=4, normalize_row=False, filename=None):
    rank = H.shape[0]
    if normalize_row:
        H /= np.linalg.norm(H, axis=1, keepdims=True)
    H3d = H.reshape(-1, img_size[0], img_size[1], order='F')
    num_cols = int(np.ceil(rank / num_rows))
    if num_rows > 1:
        fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
        cnt = 0
        for i in range(num_rows):
            for j in range(num_cols):
                if cnt < rank:
                    img = axs[i, j].imshow(H3d[cnt, :, :], cmap='gray')
                    axs[i, j].set_title(f'$h^{{{cnt + 1}}}$', fontsize=fontsize)
                    axs[i, j].axis('off')
                    divider = make_axes_locatable(axs[i, j])
                    cax = divider.append_axes('right', size='5%', pad=0.1)
                    fig.colorbar(img, cax=cax, orientation='vertical')
                else:
                    axs[i, j].axis('off')
                cnt += 1
    else:
        fig, axs = plt.subplots(1, rank, figsize=figsize)
        cnt = 0
        while cnt < rank:
            img = axs[cnt].imshow(H3d[cnt, :, :], cmap='gray')
            axs[cnt].set_title(f'$h^{{{cnt + 1}}}$', fontsize=fontsize)
            axs[cnt].axis('off')
            divider = make_axes_locatable(axs[cnt])
            cax = divider.append_axes('right', size='5%', pad=0.1)
            fig.colorbar(img, cax=cax, orientation='vertical')

            cnt += 1
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        plt.close()


def plot_combined_H(H, img_size, figsize, num_rows=1, normalize_row=False, filename=None):
    if normalize_row:
        H /= np.linalg.norm(H, axis=1, keepdims=True)

    H3d = H.reshape(-1, img_size[0], img_size[1], order='F')

    if num_rows > 1:
        num_cols = int(np.ceil(H.shape[0] / num_rows))
        large_mat = np.vstack([np.hstack(H3d[i * num_cols: (i+1) * num_cols]) for i in range(num_rows)])
    else:
        large_mat = np.hstack(H3d)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(large_mat, cmap='gray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.1)
    fig.colorbar(im, cax=cax, orientation='vertical')

    plt.tight_layout()

    if filename:
        plt.savefig(filename)
        plt.close()


def plot_W_mats(W, figsize, fontsize, n_rows=1, filename=None, scale_y=False, plot_title=None):
    rank = W.shape[1]
    wmin, wmax = np.min(W), np.max(W)

    n_cols = int(np.ceil(rank / n_rows))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.ravel()

    for cnt, ax in enumerate(axs):
        if cnt < rank:
            ax.plot(W[:, cnt], linewidth=3)
            if scale_y:
                ax.set_ylim([min(0, wmin), wmax])

            ax.set_title(f'$w_{{{cnt + 1}}}$', fontsize=fontsize)
            ax.set_xlabel('Bands')
            ax.set_ylabel('Reflectance')
        else:
            ax.axis('off')

    plt.tight_layout()
    if plot_title:
        fig.suptitle(plot_title, fontsize=25)
    if filename:
        fig.savefig(filename)
        plt.close()


def merge_images(images_list, filename, delete_images=False):
    imgs = [Image.open(i) for i in images_list]
    min_img_width = min(i.width for i in imgs)

    total_height = 0
    for i, img in enumerate(imgs):
        if img.width > min_img_width:
            imgs[i] = img.resize((min_img_width, int(img.height / img.width * min_img_width)), Image.ANTIALIAS)
        total_height += imgs[i].height

    img_merge = Image.new(imgs[0].mode, (min_img_width, total_height))
    y = 0
    for img in imgs:
        img_merge.paste(img, (0, y))
        y += img.height

    img_merge.save(filename)

    if delete_images:
        for fp in images_list:
            os.remove(fp)


def plot_and_merge(W, H, imgsize, figsize, fontsize, filenames, num_rows, delete=False):
    plot_W_mats(W, figsize, fontsize=fontsize, n_rows=num_rows, filename=filenames[0])
    plot_separate_H(H, imgsize, figsize=figsize, fontsize=fontsize, num_rows=num_rows, filename=filenames[1])
    plot_combined_H(H, imgsize, figsize=figsize, num_rows=num_rows, filename=filenames[2])
    merge_images(filenames[:3], filenames[3], delete_images=delete)
