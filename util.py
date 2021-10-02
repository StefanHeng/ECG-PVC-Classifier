"""
Utility functions
"""

import numpy as np
import pywt

import matplotlib.pyplot as plt
import seaborn as sns

from icecream import ic


def _set_sns_style():
    sns.set_style('dark')
    # sns.set_style('ticks')


_set_sns_style()
gray = list(map(lambda x: x / (2 ** 8), (128,) * 3))

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


def str_contains(arr, s):
    """
    :param arr: ndarray of string
    :param s: a string
    :return: Indices where the array element contains `s`
    """
    return np.flatnonzero(np.core.defchararray.find(arr, s) != -1)


def plot_single(arr, label):
    """ Plot single variate single dimension signal """
    plt.figure(figsize=(18, 6))
    plt.plot(np.arange(arr.size), arr, label=f'Signal {label}', marker='o', markersize=0.3, linewidth=0.25)

    handles, labels = plt.gca().get_legend_handles_labels()  # Distinct labels
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()


def plot_ecg(arr, title=None):
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

    t = 'ECG 12-lead plot'
    if title:
        t = f'{t}, {title}'
    plt.title(t)
    plt.xlabel('Time, normalized')
    plt.ylabel('Volt, normalized')
    plt.yticks(ylb_ori, ylb_new)
    handles, labels = plt.gca().get_legend_handles_labels()  # Distinct labels
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1))
    plt.show()


def plot_ecg_img(arr, low=None, hi=None, title=None, save=False):
    """
    :param arr: 1D array of ECG signal
    :param low: The minimum value for normalization across all signals
    :param hi: The maximum value for normalization across all signals
    :param title: Plot title
    :param save: Save-to-file flag

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
    t = 'ECG as Image'
    if title:
        t = f'{t}, {title}'
    plt.title(t)
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
    plt.ylabel('Energy level, normalized')
    plt.legend()
    plt.show()


def normalize_signal(arr, level=2**10):
    energies = np.sum(np.square(arr), axis=-1)[:, None]
    return (arr * level) / np.sqrt(energies)


def k_idxs(arr, k=10, d='max'):
    """
    :param arr: 1D Array
    :param k: Number of indices
    :param d: Direction, either 'max' or 'min'
    :return: Indices in the extreme
    """
    idxs = np.argpartition(arr, -k)[-k:] if d == 'max' else np.argpartition(arr, k)[:k]
    return sorted(idxs, key=lambda x: arr[x])


def plot_max_min(arr, k=10, title=None, cross_ch=False):
    """
    :param arr: 2D array
    :param k: Number of signals with large range to highlight
    :param title: Title of plot
    :param cross_ch: If true, compute range across all leads; otherwise compute each lead independently

    Illustrates the maximum and minimum globally and of each 1D array
    """
    n = arr.shape[0]
    x = np.arange(n)
    if cross_ch:
        ma = np.max(arr, axis=-1)
        mi = np.min(arr, axis=-1)
        ran = ma - mi
    else:
        arr_r = arr.reshape(n, N_LD, -1)
        ma = np.max(arr_r, axis=-1)
        mi = np.min(arr_r, axis=-1)
        ran = np.max(ma - mi, axis=-1)
        ma = np.max(ma, axis=-1)
        mi = np.min(mi, axis=-1)

    fig, axs = plt.subplots(2, figsize=(20, 9))
    fig.tight_layout(pad=5)

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
    axs[0].set_xlabel('Entry no.')
    axs[0].set_ylabel('Energy level, normalized')
    axs[1].set_title('Range')
    axs[1].set_xlabel('Entry no.')
    axs[1].set_ylabel('Energy level, normalized')
    t = 'Max and Min plot'
    if title:
        t = f'{t}, {title}'
    plt.suptitle(t)
    plt.show()


def plot_mean(arr, title=None):
    n = arr.shape[0]
    x = np.arange(n)
    avg = np.mean(arr, axis=-1)

    plt.figure(figsize=(20, 9), constrained_layout=True)
    plt.plot(x, avg, marker='o', ms=0.3, lw=0.25, label='Average volt')
    plt.fill_between(x, avg, np.full(n, 0), alpha=0.3)

    t = 'Mean Volt by entry'
    if title:
        t = f'{t}, {title}'
    plt.title(t)
    plt.xlabel('Entry no.')
    plt.ylabel('Volt, normalized')
    plt.legend()
    plt.show()


def plot_wavelet_dwt(arr, w, level=6, title=None):
    """Show multi-level dwt coefficients for given data and wavelet """
    cfs = pywt.wavedec(arr, w, mode='smooth', level=level)
    appr, dtls = cfs[0], cfs[1:]
    n_plot = level+2  # Original and 7 decompositions
    cs = iter(sns.color_palette(palette='husl', n_colors=n_plot))

    fig = plt.figure(figsize=(2 * 4 * 2, len(dtls) * 2), constrained_layout=True)
    subfigs = fig.subfigures(1, 2)

    def _plot(idx_fig, idx_plt, y, label, title_=None):
        ax = subfigs[idx_fig].add_subplot(*idx_plt)
        ax.plot(np.arange(y.size), y, marker='o', ms=0.3, c=next(cs), lw=0.25, label=label)
        ax.axhline(y=0, lw=0.4, c=gray)
        if title_:
            ax.set_title(title_)
        ax.legend()

    def _c(s):
        return f'{s} coefficients'

    n_row, n_col = level, 1

    def _idx_plt(i_plt):
        return [n_row, n_col, i_plt]
    o = 'Original'
    a = 'Approximation'
    d = 'Detail'
    _plot(0, _idx_plt(1), arr, o, title_=o)
    _plot(0, _idx_plt(n_row), appr, _c(a), title_=a)

    for idx, dtl in enumerate(reversed(dtls)):
        _plot(1, _idx_plt(idx+1), dtl, f'{_c(d)} - {idx+1}', title_=d)

    t = 'Wavelet Multilevel Decomposition'
    if title:
        t = f'{t}, {title}'
    plt.suptitle(t)
    plt.show()


def plot_wavelet_cwt(arr, w, max_scale=128, title=None):
    plt.figure(figsize=(8 * (arr.shape[0] / max_scale), 8), constrained_layout=True)
    scales = np.arange(1, max_scale)
    cfs, fqs = pywt.cwt(arr, scales, w)
    cfs = np.abs(cfs)
    with sns.axes_style('ticks'):
        sns.heatmap(cfs, xticklabels=10, yticklabels=5, square=True, cmap='mako')
    t = 'Wavelet Power Spectrum'
    if title:
        t = f'{t}, {title}'
    plt.title(t)

    plt.yticks(rotation=0)
    plt.ylabel('Scale')
    plt.xlabel('Time')
    plt.show()


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
    # plot_max_min(ecgs_norm, k=5, cross_ch=False)
    # plot_mean(ecgs_norm)

    data = ecgs_norm[502][:350]
    # plot_wavelet_dwt(data, 'db2', level=6, title='Example')
    plot_wavelet_cwt(data, 'cmor1.5-1.0', max_scale=128, title='Example')
