"""
Utility functions
"""

import numpy as np

import matplotlib.pyplot as plt
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


def normalize_signal(arr, level=2**10):
    energies = np.sum(np.square(arr), axis=-1)[:, None]
    return (arr * level) / np.sqrt(energies)

