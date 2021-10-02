import numpy as np
import pandas as pd

import re
import h5py

import os
import scipy.io

import seaborn as sns
import matplotlib.pyplot as plt
from icecream import ic

from ecg_data import DataGetter


if __name__ == '__main__':

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

    # segments = 10
    # points_per_segment = 100
    #
    # # your data preparation will vary
    # x = np.tile(np.arange(points_per_segment), segments)
    # z = np.floor(np.arange(points_per_segment * segments) / points_per_segment)
    # y = np.sin(x * (1 + z))
    #
    # df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    #
    # pal = sns.color_palette()
    # g = sns.FacetGrid(df, row="z", hue="z", aspect=15, height=.5, palette=pal)
    # g.map(plt.plot, 'x', 'y')
    # g.map(plt.axhline, y=0, lw=2, clip_on=False)
    # # Set the subplots to overlap
    # g.fig.subplots_adjust(hspace=-.1)
    # g.set_titles("")
    # g.set(yticks=[])
    # g.despine(bottom=True, left=True)
    #
    # plt.show()


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

    # a = np.array([
    #     [1, 3],
    #     [2, 5]
    # ])
    # es = np.sum(np.square(a), axis=-1)
    # ic(es)
    # a_norm = a / np.sqrt(es[:, None])
    # ic(a_norm, np.sum(np.square(a_norm), axis=-1))

    # a = 'asdhjlasd'
    # for i in a:
    #     ic(i)

    # np.random.seed(7)
    #
    # ic(np.random.randint(20, size=10))

    # import seaborn as sns
    # ic(sns.color_palette())
    # plt.stem([0, 2, 1])
    # plt.show()

    # ic(4 % 1)

    # def str_contains(arr, s):
    #     return np.flatnonzero(np.core.defchararray.find(arr, s) != -1)
    #
    # a = np.array(['as', 'vd', 'assa'])
    # ic(a)
    # ic(np.where('as' in a))
    # ic(np.where('a' in a))
    # ic(np.where(a == 2))
    # ic(str_contains(a, 'a'))

    # a = np.array([
    #     ['as', 'b'],
    #     ['cc', 's']
    # ], dtype='S')
    # ic(a)

    # fl = h5py.File('file', 'w')
    #
    # d = pd.DataFrame({'float': [1.0],
    #                   'int': [1],
    #                   'datetime': [pd.Timestamp('20180310')],
    #                   'string': ['foo']})
    # ic(d.dtypes)
    # a = d['string']
    # ic(a.dtype)
    # a = a.astype('string')
    # a = a.astype('S')
    # ic(a)
    # d['string'] = a
    # ic(d)
    # # df.to_hdf('data.h5', key='df', mode='w')
    # # fl.attrs['dae'] = d
    # # fl.create_dataset('a', data=d)
    #
    # # store = pd.HDFStore('store.h5')
    # #
    # # store['df'] = d  # save it
    # # ic(store['df'])
    #
    # fl['a'] = d.columns.values.astype('S')
    # ic(d.dtypes)
    #
    # # fl['b'] = d
    # df2 = d.astype('S')
    # ic(df2.dtypes)
    #
    # a = np.array([
    #     ['as', 'b'],
    #     ['cc', 's']
    # ], dtype='S')
    #
    # fl['b'] = a
    #
    # ic(fl)
    # ic(fl['a'])
    # ic(fl['b'])

    # a = np.array(['1', '2'])
    # ic(a.astype('int'))

    df = pd.DataFrame([[4, 9]] * 3, columns=['A', 'B'])

    def _map(x):
        ic(x[0])
        ic(x['B'])

    df.apply(_map, axis=1, result_type='expand')

    if type(1.0) is float:
        ic('asd')





