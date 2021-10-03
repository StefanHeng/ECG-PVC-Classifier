import numpy as np
import pandas as pd
import h5py
from icecream import ic

from util import *


class EcgLabel:
    """
    Manages the labels corresponding to ECG signals as in EcgData

    A wrapper for `pandas.DataFrame`
    """
    F = 'ecg_label.hdf5'

    def __init__(self, dnm: str):
        self.dnm = dnm
        rec = h5py.File(self.F, 'r')[dnm]
        columns = rec['columns'][:].astype('str')  # Per `EcgLabelExport`, bytes to strings
        dtypes = rec['dtypes'][:].astype('str')
        d = rec['data'][:].astype('str')
        self.df: pd.DataFrame = pd.DataFrame(d, columns=columns)
        for dtp in dtypes:
            self.df = self.df.astype(dict(zip(columns, dtp)))

    def __getitem__(self, k):
        return self.df[k]

    def unique(self):
        """ Returns a dataframe with unique rows based on the set of labels """
        d_set = dict(
            daeVt=['wall', 'origin', 'site']
        )
        return self.df.drop_duplicates(subset=d_set[self.dnm])


if __name__ == '__main__':
    import os

    os.chdir('../PVC_DATA')

    el = EcgLabel('daeVt')
    ic(el.df)
    ic(el.df.dtypes)
    ic(el.unique())
    # ic(el['pat_num'])

