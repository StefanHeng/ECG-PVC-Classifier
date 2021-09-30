import matplotlib.pyplot as plt
import seaborn as sns

from icecream import ic

from util import *


if __name__ == '__main__':
    plot_palette()

    for pnm in ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']:
        plot_palette(pnm)

    for pnm in ['Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3']:
        plot_palette(pnm)
