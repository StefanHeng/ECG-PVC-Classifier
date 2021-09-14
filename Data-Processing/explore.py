import numpy as np

import os
import scipy.io

from ecg_data import DataGetter

from icecream import ic


def pprint(m):
    keys = list(filter(lambda x: not x.startswith('__') and not x.endswith('__'), m.keys()))
    ic(keys)

    for k in keys:
        d = m[k]
        ic(k, d.shape, d)


if __name__ == '__main__':
    os.chdir('../../PVC_DATA')

    # mat = scipy.io.loadmat('MM_classifier/vtlist_matlab_dyj.mat')
    # pprint(mat)
    #
    # # # dset = DataGetter()('daeRawCnm')
    # dset = DataGetter()('daeVtLst')
    # dset.overview()

    # Content of all files at glance
    for k in DataGetter.Dataset.D:
        DataGetter()(k).overview()
