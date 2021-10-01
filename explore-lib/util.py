from typing import Union
from copy import deepcopy
from math import ceil

import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic

import pywt

sns.set_style('dark')
gray = list(map(lambda x: x / (2 ** 8), (128,) * 3))
# Taken from https://medium.com/@morganjonesartist/color-guide-to-seaborn-palettes-da849406d44f
SNS_CMAPS = [
    'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r',
    'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd',
    'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1',
    'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r',
    'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r',
    'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds',
    'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r',
    'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd',
    'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone',
    'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r',
    'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r',
    'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg',
    'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r',
    'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno',
    'inferno_r', 'magma', 'magma_r', 'mako', 'mako_r',
    'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r',
    'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r',
    'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r',
    'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b',
    'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'twilight',
    'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'
]


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

    for pnm in ['Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3', 'tab10']:
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
