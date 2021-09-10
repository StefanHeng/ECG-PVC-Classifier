# import numpy as np

import os
from scipy.io import loadmat

import re
from typing import Union

from icecream import ic

os.chdir('../PVC_DATA')


class DataGetter:
    """
    Interface to fetch ECG data from .MAT files in the datasets
    """
    class Dataset:
        """
        Interface to MAT file dictionary output
        """
        D = dict(
            dae=dict(
                path='MM_classifier/vtdata.mat',
                keys=dict(
                    ecg='vt_ecg',
                    nm='vtname',
                    cnm='vtcasename',
                    qrs_width='vt_qrswidth'
                )
            ),
        )

        def __init__(self, d_nm: str):
            """
            :param d_nm: Database code
            """
            attrs = self.D[d_nm]
            mat = loadmat(attrs['path'])
            key_map = attrs['keys']
            self.d = {}
            self.keys = key_map.keys()
            for k, v in key_map.items():
                self.d[k] = mat[v]

            for k, data in self.d.items():
                if 1 in data.shape:
                    self.d[k] = data.reshape(data.size)

        def __getitem__(self, k):
            """
            Horizontal & vertical indexing
            """
            if k in self.keys:
                return self.d[k]
            else:  # Expect int
                if type(k) is str and re.match(r'[0-9]+', k):
                    k = int(k)
                d = {}
                for keys in self.keys:
                    d[keys] = self.__getitem__(keys)[k]
                for k, v in d.items():
                    if v.dtype.type is np.str_:
                        d[k] = v[0].removesuffix('.mat')
                return d

    def get(self, k: Union[str, list]):
        """
        :param k: Key, determines which data to fetch. Expect a dot(`.`)-separated string or a list

        The 1st string specifies which dataset, If only 1 code is provided, the entire dictionary is returned
        The 2nd string or integer, if provided, returns the corresponding key in the data set

        The hierarchy of valid codes: {
            - 'dae': Based on `Data for Dae and Weiqing_myver`, n=156
                - 'ecg': The ECG signal
                - 'qrs_width': The width of QRS complex
                - 'nm': Name
                - 'cnm': Case name
        }
        """
        codes = k.split('.') if type(k) is str else k
        d_nm = codes[0]
        dset = self.Dataset(d_nm)
        if len(codes) == 1:
            return dset
        else:
            c = codes[1]
            return dset[c]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from util import *

    dg = DataGetter()

    idx = 20
    dataset = dg.get('dae')
    ecg = dg.get(['dae', 'ecg'])
    nm = dg.get('dae.nm')
    cnm = dataset['cnm']
    width = dataset['qrs_width']

    for i in [ecg, nm, cnm, width]:
        ic(i[:10], i.shape)

    ds = [
        dataset[str(idx)],
        dataset[idx + 1],
        dg.get(['dae', idx + 2]),
        dg.get(['dae', f'{idx + 3}'])
    ]
    ic(ds[0])
    for ecg_d in ds:
        arr = ecg_d['ecg']
        nm = ecg_d['cnm']
        plot_single(arr, nm)
