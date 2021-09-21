"""
Utility functions
"""

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from icecream import ic


N_LD = 12


def sizeof_fmt(n):
    """
    Converts byte size to human readable format
    :param n: Number of bytes
    """
    suffix = 'B'

    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(n) < 1024.0:
            return "%3.1f%s%s" % (n, unit, suffix)
        n /= 1024.0
    return "%.1f%s%s" % (n, 'Yi', suffix)


def plot_single(arr, label):
    plt.figure(figsize=(18, 6))
    plt.plot(np.arange(arr.size), arr, label=f'Signal {label}', marker='o', markersize=0.3, linewidth=0.25)

    handles, labels = plt.gca().get_legend_handles_labels()  # Distinct labels
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()


def plot_ecg(arr, title='ECG 12-lead plot'):
    """
    Assumes arr is concatenated 1D array of 12-lead signals
    """
    n = N_LD
    height = (abs(np.max(arr)) + abs(np.min(arr))) / 4  # Empirical

    arr = arr.reshape(n, -1)
    plt.figure(figsize=(5, 13), constrained_layout=True)

    ylb_ori = ((np.arange(n) - n + 1) * height)[::-1]
    ylb_new = ['I', 'II', 'III', 'avR', 'avL', 'avF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    for i, row in enumerate(arr):
        offset = height * i
        x = np.arange(row.size)
        y = row - offset
        plt.plot(x, y, label=ylb_new[i], marker='o', markersize=0.3, linewidth=0.25)
        plt.axhline(y=-offset, lw=0.2)

    plt.title(title)
    plt.yticks(ylb_ori, ylb_new)
    handles, labels = plt.gca().get_legend_handles_labels()  # Distinct labels
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1))
    plt.show()


def plot_ecg_img(arr, low=None, hi=None, title='ECG as image plot', save=False):
    """
    :param arr: 1D array of ECG signal
    :param low: The minimum value for normalization across all signals
    :param hi: The maximum value for normalization across all signals
    :param title: Plot title

    If `low` and `hi` unspecified, will normalize based on `arr`
    """
    if low is None and hi is None:
        low = np.min(arr)
        hi = np.max(arr)

    arr -= low
    arr *= 255.0 / (hi - low)
    arr = arr.reshape(N_LD, -1)

    plt.figure(figsize=(12, 1), constrained_layout=True)
    plt.imshow(arr, interpolation='nearest', cmap='gray', vmin=0, vmax=255)
    plt.title(title)

    if save:
        plt.savefig(f'{title}.png', dpi=300)
    plt.show()


def plot_energy(arr, title=None):
    x = np.arange(arr.shape[0])
    energy = np.sum(np.square(arr), axis=-1)

    plt.figure(figsize=(20, 9), constrained_layout=True)
    plt.plot(x, energy, marker='o', ms=0.3, lw=0.25, label='Energy level')
    plt.fill_between(x, energy, alpha=0.3)

    t = 'Energy level by entry'
    if title:
        t = f'{t}, {title}'
    plt.title(t)
    plt.xlabel('Entry no.')
    plt.ylabel('Energy level')
    plt.legend()
    plt.show()


def k_idxs(arr, k=10, d='max'):
    """
    :param arr: 1D Array
    :param k: Number of indices
    :param d: Direction, either 'max' or 'min'
    :return: Indices in the extreme
    """
    idxs = np.argpartition(arr, -k)[-k:] if d == 'max' else np.argpartition(arr, k)[:k]
    return sorted(idxs, key=lambda x: arr[x])


def plot_max_min(arr, k=10, title=None):
    """
    :param arr: 2D array
    :param k: Number of signals with large range to highlight

    Illustrates the maximum and minimum globally and of each 1D array
    """
    x = np.arange(arr.shape[0])
    ma = np.max(arr, axis=-1)
    mi = np.min(arr, axis=-1)
    ran = ma - mi

    fig, axs = plt.subplots(2, figsize=(20, 9), constrained_layout=True)
    cs = sns.color_palette()
    axs[0].plot(x, ma, marker='o', ms=0.3, lw=0.25, c=cs[0], label='Local max')
    axs[0].axhline(y=np.max(ma), lw=0.4, c=cs[0], label='Global max')
    axs[0].plot(x, mi, marker='o', ms=0.3, lw=0.25, c=cs[1], label='Local min')
    axs[0].axhline(y=np.min(mi), lw=0.4, c=cs[1], label='Global min')
    axs[0].fill_between(x, ma, mi, fc=cs[2], alpha=0.3)

    axs[1].plot(x, ran, marker='o', ms=0.3, lw=0.25, c=cs[4], label='Range')
    axs[1].axhline(y=np.max(ran), lw=0.4, c=cs[4], label='Max')
    axs[1].axhline(y=np.min(ran), lw=0.4, c=cs[4], label='Min')
    axs[1].fill_between(x, ran, fc=cs[4], alpha=0.3)

    for idx in k_idxs(ran, k=k, d='max'):
        axs[0].axvline(x=idx, lw=0.4, c=cs[3], label='K max')
        axs[1].axvline(x=idx, lw=0.4, c=cs[3], label='K max')
    for idx in k_idxs(ran, k=k, d='min'):
        axs[0].axvline(x=idx, lw=0.4, c=cs[5], label='K min')
        axs[1].axvline(x=idx, lw=0.4, c=cs[5], label='K min')

    handles, labels = plt.gca().get_legend_handles_labels()  # Distinct labels
    by_label = dict(zip(labels, handles))
    axs[0].legend(by_label.values(), by_label.keys())
    axs[1].legend(by_label.values(), by_label.keys())
    axs[0].set_title('Max and Min')
    axs[1].set_title('Range')
    t = 'Max and Min plot'
    if title:
        t = f'{t}, {title}'
    plt.suptitle(t)
    plt.show()


def normalize_signal(arr, level=2**10):
    energies = np.sum(np.square(arr), axis=-1)[:, None]
    return (arr * level) / np.sqrt(energies)


if __name__ == '__main__':
    import numpy as np

    import os

    from icecream import ic

    from ecg_data import DataGetter

    os.chdir('../PVC_DATA')

    dg = DataGetter()
    ecgs = dg('daePm.ecg')
    ecgs_norm = normalize_signal(ecgs)
    # plot_energy(ecgs)
    plot_max_min(ecgs, k=5)



