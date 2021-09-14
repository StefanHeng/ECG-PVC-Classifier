import numpy as np

import re

import os
import scipy.io

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

    a = np.array([
        np.array(['66_VT1.mat'], dtype='<U10'),
        np.array(['67_VT1.mat'], dtype='<U10'),
        np.array(['67_VT1_2.mat'], dtype='<U12')
    ])
    ic(a, a.dtype.type, a.shape)

    x = np.array([u'12345', u'abc'], dtype=object)
    ic(x, x.shape, x.dtype)
    x = x.astype(str)
    ic(x)

    ic(re.match(r'([0-9]+)_', '66_VT1.mat').groups()[0])
    ic(a[0])
    ic(np.vectorize(lambda s: re.match(r'([0-9]+)_', s).groups()[0])(a))

    arr = DataGetter()('daeRaw.vt_case_name')
    ic(type(arr[0][0]), arr[0][0], arr[0][0].dtype.type is np.str_)
    ic(arr.dtype is np.dtype('O'))

    b = np.vectorize(lambda x: x[0])(arr)
    ic(b)
    ic(arr.astype('U'))

    arr = arr.reshape(-1, 1)
    ic(arr.shape, arr.dtype, arr[:20])
    c = arr.astype('U')
    ic(c)
    b = arr.astype(np.str_)
    ic(arr[0])

    ic(np.vectorize(lambda s: re.match(r'([0-9]+)_', s).groups()[0])(arr))


