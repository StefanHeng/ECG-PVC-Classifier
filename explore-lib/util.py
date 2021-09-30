from typing import Union
from copy import deepcopy
from math import ceil

import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic

import pywt

sns.set_style("dark")
gray = list(map(lambda x: x / (2 ** 8), (128,) * 3))


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

    _plot(a)
    if n_colors:
        for n in n_colors:
            a_ = deepcopy(a)
            a_['n_colors'] = n
            _plot(a_, n)


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
    # if f in ['shan', 'fbsp', 'cmor']:
    #     wnames = list(map(lambda x: f'{x}1.5-1.0', wnames))  # Given warning

    cont = wnames[0] in continuous
    Obj = pywt.ContinuousWavelet if cont else pywt.Wavelet
    w_dummy = Obj(wnames[0])
    n_func = len(w_dummy.wavefun()) - 1  # Takes on [1, 2, 4]
    cont_cpx = cont and w_dummy.complex_cwt
    if cont_cpx:
        n_func *= 2
    ic(f)
    bio = n_func == 4

    l = len(wnames)
    wnames = iter(wnames)
    cs = iter(sns.color_palette(palette='husl', n_colors=l))

    n_col = 4
    n_row = ceil(l / (n_col // n_func))
    fig = plt.figure(figsize=(n_col * 4 * 2, n_row * 2), constrained_layout=True)

    for i in range(l):
        r, c = i // (n_col // n_func), i % (n_col // n_func)
        if n_func == 1:
            idx = 1 + r * n_col + c
        else:
            idx = 1 + (r * n_func + c) * (n_col // n_func)
        w = Obj(next(wnames))
        clr = next(cs)

        def _plot(idx_, y, label):
            ax = fig.add_subplot(n_row, n_col, idx_)
            ax.set_title(w.name)
            ax.plot(x, y, marker='o', ms=0.3, c=clr, lw=0.25, label=label)
            ax.axhline(y=0, lw=0.4, c=gray)
            ax.legend()

        if cont and not cont_cpx:
            psi, x = w.wavefun(level=level)
            _plot(idx, psi, 'Scaling func')

        elif not bio or cont_cpx:
            if cont_cpx:
                psi, x = w.wavefun(level=level)
                _plot(idx, psi.real, 'Scaling func, real')
                _plot(idx+1, psi.imag, 'Scaling func, imag')
            else:
                phi, psi, x = w.wavefun(level=level)
                _plot(idx, phi, 'Scaling func')
                _plot(idx+1, psi, 'Wavelet func')
        else:
            phi_d, psi_d, phi_r, psi_r, x = w.wavefun(level=level)
            _plot(idx, phi_d, 'Scaling func, decomp')
            _plot(idx+1, psi_d, 'Wavelet func, decomp')
            _plot(idx+2, phi_d, 'Scaling func, recomp')
            _plot(idx+3, psi_d, 'Scaling func, recomp')
    plt.show()


if __name__ == '__main__':
    # plot_palette()
    # plt.show()

    # plot_all_palettes()
    plot_all_wavelets('shan')
