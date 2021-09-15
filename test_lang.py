import numpy as np
import pandas as pd

import re

import os
import scipy.io

import seaborn as sns
import matplotlib.pyplot as plt
from icecream import ic

from ecg_data import DataGetter


if __name__ == '__main__':
    os.chdir('../PVC_DATA')

    # mat = scipy.io.loadmat('MM_classifier/alldata.mat')
    # ic(mat, mat.keys())

    # casename = mat['vtcasename']  # An ndarray
    # ic(casename, type(casename), len(casename))
    # arr = casename[0]  # An ndarray
    # ic(arr, type(arr), len(arr))
    #
    # arr = arr[0]  # An ndarray
    # arr = arr[0]  # An ndarray
    # ic(arr, type(arr), len(arr))

    # customer_name = "John Milton"
    # ic(customer_name.startswith("J"), type(customer_name))

    # a = np.array([-8903.65, -8831.65, -8747.65, np.nan, np.nan, np.nan])
    # ic(a)
    # b = a == np.nan
    # ic(b)
    #
    # ic(np.isnan(a))
    #
    # class A:
    #     def __init__(self, i):
    #         self.i = i
    #         if i > 0:
    #             ic()
    #             self.a = A(i - 1)
    #
    # ic(A(1))

    # a = np.array([
    #     np.array(['66_VT1.mat'], dtype='<U10'),
    #     np.array(['67_VT1.mat'], dtype='<U10'),
    #     np.array(['67_VT1_2.mat'], dtype='<U12')
    # ])
    # ic(a, a.dtype.type, a.shape)
    #
    # x = np.array([u'12345', u'abc'], dtype=object)
    # ic(x, x.shape, x.dtype)
    # x = x.astype(str)
    # ic(x)
    #
    # ic(re.match(r'([0-9]+)_', '66_VT1.mat').groups()[0])
    # ic(a[0])
    # ic(np.vectorize(lambda s: re.match(r'([0-9]+)_', s).groups()[0])(a))
    #
    # arr = DataGetter()('daeRaw.vt_case_name')
    # ic(type(arr[0][0]), arr[0][0], arr[0][0].dtype.type is np.str_)
    # ic(arr.dtype is np.dtype('O'))
    #
    # b = np.vectorize(lambda x: x[0])(arr)
    # ic(b)
    # ic(arr.astype('U'))
    #
    # arr = arr.reshape(-1, 1)
    # ic(arr.shape, arr.dtype, arr[:20])
    # c = arr.astype('U')
    # ic(c)
    # b = arr.astype(np.str_)
    # ic(arr[0])
    #
    # ic(np.vectorize(lambda s: re.match(r'([0-9]+)_', s).groups()[0])(arr))

    segments = 10
    points_per_segment = 100

    # your data preparation will vary
    x = np.tile(np.arange(points_per_segment), segments)
    z = np.floor(np.arange(points_per_segment * segments) / points_per_segment)
    y = np.sin(x * (1 + z))

    df = pd.DataFrame({'x': x, 'y': y, 'z': z})

    pal = sns.color_palette()
    g = sns.FacetGrid(df, row="z", hue="z", aspect=15, height=.5, palette=pal)
    g.map(plt.plot, 'x', 'y')
    g.map(plt.axhline, y=0, lw=2, clip_on=False)
    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.1)
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    plt.show()


    # # plot 1:
    # x = np.array([0, 1, 2, 3])
    # y = np.array([3, 8, 1, 10])
    #
    # plt.subplot(2, 1, 1)
    # plt.plot(x, y)
    #
    # # plot 2:
    # x = np.array([0, 1, 2, 3])
    # y = np.array([10, 20, 30, 40])
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(x, y)
    # plt.subplots_adjust(hspace=-0.2)
    #
    # plt.show()


