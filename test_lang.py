import numpy as np

import os
import scipy.io

from icecream import ic


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

    a = np.array([-8903.65, -8831.65, -8747.65, np.nan, np.nan, np.nan])
    ic(a)
    b = a == np.nan
    ic(b)

    ic(np.isnan(a))

    class A:
        def __init__(self, i):
            self.i = i
            if i > 0:
                ic()
                self.a = A(i - 1)

    ic(A(1))

