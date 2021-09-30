import itertools
from typing import Union
from copy import deepcopy
from math import ceil

import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic

import pywt


def plot_palette(pnm=None, title=None, n_colors=None):
    a = dict()
    if pnm:
        a['palette'] = pnm

    if title:
        t = title
    elif pnm:
        t = pnm
    else:
        t = 'Default'

    def _plot(args, num=None):
        p = sns.color_palette(**args)
        if not num:
            num = len(p)
        sns.palplot(p)
        plt.title(f'{t} - {num}')

    if n_colors:
        for n in n_colors:
            a_ = deepcopy(a)
            a_['n_colors'] = n
            _plot(a_, n)
    else:
        _plot(a)


def plot_all_palettes(n_colors: Union[list, None] = None):
    plot_palette(n_colors=n_colors)

    for pnm in ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']:
        plot_palette(pnm=pnm, n_colors=n_colors)

    for pnm in ['Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3']:
        plot_palette(pnm=pnm, n_colors=n_colors)

    for pnm in ['hls', 'husl']:
        plot_palette(pnm=pnm, n_colors=n_colors)

    plt.show()


continuous = set(pywt.wavelist(kind='continuous'))


def plot_all_wavelets(f, level=5):
    """
    Taken from `pywt/demo/plot_wavelets.py`

    :param f: Wavelet family
    :param level:
    """
    wnames = pywt.wavelist(f)
    cont = wnames[0] in continuous
    Obj = pywt.ContinuousWavelet if cont else pywt.Wavelet
    n_func = len(Obj(wnames[0]).wavefun()) - 1  # Either 2 or 4
    bio = n_func == 4
    l = len(wnames)
    wnames = iter(wnames)
    cs = iter(sns.color_palette(palette='husl', n_colors=l))

    n_col = 4
    n_row = ceil(l / (n_col // n_func))
    fig = plt.figure(figsize=(n_col * 4 * 2, n_row * 2), constrained_layout=True)

    for i in range(l):
        r, c = i // (n_col // n_func), i % (n_col // n_func)
        idx = 1 + (r * n_func + c) * (n_col // n_func)
        w = Obj(next(wnames))
        clr = next(cs)

        def _plot(idx_, y, label):
            ax = fig.add_subplot(n_row, n_col, idx_)
            ax.set_title(w.name)
            ax.plot(x, y, marker='o', ms=0.3, c=clr, lw=0.25, label=label)
            ax.axhline(y=0, lw=0.4)
            ax.legend()

        if bio:
            phi_d, psi_d, phi_r, psi_r, x = w.wavefun(level=level)
            _plot(idx, phi_d, 'Scaling func, decomp')
            _plot(idx+1, psi_d, 'Wavelet func, decomp')
            _plot(idx+2, phi_d, 'Scaling func, recomp')
            _plot(idx+3, psi_d, 'Scaling func, recomp')
        else:
            phi, psi, x = w.wavefun(level=level)
            _plot(idx, phi, 'Scaling func')
            _plot(idx+1, phi, 'Wavelet func')

    plt.show()
    # return
    #
    # plot_data = [('db', (4, 3)),
    #              ('sym', (4, 3)),
    #              ('coif', (3, 2))]
    #
    # for family, (rows, cols) in plot_data:
    #     fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    #     # fig.subplots_adjust(hspace=0.2, wspace=0.2, bottom=.02, left=.06, right=.97, top=.94)
    #     colors = itertools.cycle('bgrcmyk')
    #
    #     wnames = pywt.wavelist(family)
    #     i = iter(wnames)
    #     for col in range(cols):
    #         for row in range(rows):
    #             try:
    #                 wavelet = pywt.Wavelet(next(i))
    #             except StopIteration:
    #                 break
    #             phi, psi, x = wavelet.wavefun(level=5)
    #
    #             color = next(colors)
    #             ax = fig.add_subplot(rows, 2 * cols, 1 + 2 * (col + row * cols))
    #             ax.set_title(wavelet.name + " phi")
    #             ax.plot(x, phi, color)
    #             ax.set_xlim(min(x), max(x))
    #
    #             ax = fig.add_subplot(rows, 2 * cols, 1 + 2 * (col + row * cols) + 1)
    #             ax.set_title(wavelet.name + " psi")
    #             ax.plot(x, psi, color)
    #             ax.set_xlim(min(x), max(x))
    #     break

    # for family, (rows, cols) in [('bior', (4, 3)), ('rbio', (4, 3))]:
    #     fig = plt.figure(figsize=(16, 9))
    #     fig.subplots_adjust(hspace=0.5, wspace=0.2, bottom=.02, left=.06,
    #                         right=.97, top=.94)
    #
    #     colors = itertools.cycle('bgrcmyk')
    #     wnames = pywt.wavelist(family)
    #     i = iter(wnames)
    #     for col in range(cols):
    #         for row in range(rows):
    #             try:
    #                 wavelet = pywt.Wavelet(next(i))
    #             except StopIteration:
    #                 break
    #             phi, psi, phi_r, psi_r, x = wavelet.wavefun(level=5)
    #             row *= 2
    #
    #             color = next(colors)
    #             ax = fig.add_subplot(2*rows, 2*cols, 1 + 2*(col + row*cols))
    #             ax.set_title(wavelet.name + " phi")
    #             ax.plot(x, phi, color)
    #             ax.set_xlim(min(x), max(x))
    #
    #             ax = fig.add_subplot(2*rows, 2*cols, 2*(1 + col + row*cols))
    #             ax.set_title(wavelet.name + " psi")
    #             ax.plot(x, psi, color)
    #             ax.set_xlim(min(x), max(x))
    #
    #             row += 1
    #             ax = fig.add_subplot(2*rows, 2*cols, 1 + 2*(col + row*cols))
    #             ax.set_title(wavelet.name + " phi_r")
    #             ax.plot(x, phi_r, color)
    #             ax.set_xlim(min(x), max(x))
    #
    #             ax = fig.add_subplot(2*rows, 2*cols, 1 + 2*(col + row*cols) + 1)
    #             ax.set_title(wavelet.name + " psi_r")
    #             ax.plot(x, psi_r, color)
    #             ax.set_xlim(min(x), max(x))

    plt.show()


if __name__ == '__main__':
    # plot_palette()
    # plt.show()

    # plot_all_palettes()
    # plot_all_wavelets('sym')
    # obj = pywt.ContinuousWavelet if True else pywt.Wavelet
    # w = obj('gaus1')
    # ic(w)
    plot_all_wavelets('gaus')
