import numpy as np
import pandas as pd

import os
import scipy.io

import seaborn as sns
import matplotlib.pyplot as plt
from icecream import ic

from util import *
from ecg_data import DataGetter


def pprint(m):
    keys = list(filter(lambda x: not x.startswith('__') and not x.endswith('__'), m.keys()))
    ic(keys)

    for k in keys:
        d = m[k]
        ic(k, d.shape, d)


if __name__ == '__main__':
    os.chdir('../../PVC_DATA')

    dg = DataGetter()

    # mat = scipy.io.loadmat('MM_classifier/vtlist_matlab_dyj.mat')
    # pprint(mat)
    #
    # # # dset = DataGetter()('daeRawCnm')
    # dset = DataGetter()('daeVtLst')
    # dset.overview()

    # Content of all files at glance
    for k in dg.DSETS:
        DataGetter()(k).overview()

    for dnm in dg.DSETS:
        d = dg(dnm)
        ic(dnm, d.path, d.meta)

    d = dg(['daeRaw', 0])
    arr = d['vt'][:d['vt_width']]
    assert np.count_nonzero(~np.isnan(arr)) == arr.size
    ic(np.max(arr), np.min(arr))
    ic(abs(np.max(arr)) + abs(np.min(arr)) / 100)
    plot_single(arr, 'daeRaw.0')

    d = dg(['daeVt', 0])
    arr = d['ecg']
    ic(arr.shape)
    plot_ecg(arr, 'daeVt.0')
