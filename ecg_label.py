import numpy as np
import pandas as pd
import h5py
from icecream import ic


class EcgLabel:
    """
    Manages the labels corresponding to ECG signals as in EcgData
    """
    F = 'ecg_label.hdf5'

    def __init__(self, dnm: str):
        rec = h5py.File(self.F, 'r')[dnm]
        columns = rec['columns'][:].astype('str')  # Per `EcgLabelExport`, bytes to strings
        dtypes = rec['dtypes'][:].astype('str')
        data = rec['data'][:].astype('str')
        self.df = pd.DataFrame(data, columns=columns)
        # self.df = self.df.astype({'site': 'string'})
        # self.df['site'] = self.df['site'].astype('string')
        #
        # a = self.df['site'][0]
        # ic(type(a))
        # ic(a)
        # # ic(self.df['site'].astype('string'))
        # ic(self.df['site'])
        #
        # self.df['site'] = self.df['site'].astype('category')
        # # ic(self.df['site'].astype('string'))
        # ic(self.df['site'])

        for dtp in dtypes:
            ic(list(dtp))
            self.df = self.df.astype(dict(zip(columns, dtp)))
            ic(self.df)

        # self.df = self.df.astype({'site': 'string'})
        for c in self.df.columns:
            ic(self.df[c])


if __name__ == '__main__':
    import os

    os.chdir('../PVC_DATA')

    el = EcgLabel('daeVt')

