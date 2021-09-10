"""
Utility functions
"""

import numpy as np

import matplotlib.pyplot as plt


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
