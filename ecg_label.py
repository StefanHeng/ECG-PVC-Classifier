import numpy as np
import pandas as pd
import h5py
from enum import IntEnum
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
        for ds in dtypes:
            self.df = self.df.astype(dict(zip(columns, ds)))

        for idx, dtp in enumerate(dtypes[-1]):
            if dtp == 'category' and dtypes[0][idx] == 'string':
                col_nm = columns[idx]
                ic(col_nm)
                col = self.df[columns[idx]]
                cat = self.df[columns[idx]].cat
                c = cat.categories
                # categories = self.df.iloc[:, idx].unique()
                c_ = list(c)
                if 'NA' in c_:  # `NP` as the smallest category
                    idx = c_.index('NA')
                    c_[0], c_[idx] = c_[idx], c_[0]
                    ic(c_)
                    self.df[col_nm] = col.cat.reorder_categories(c_)
                ic(c, self.df[col_nm].cat.categories)

    def __getitem__(self, k):
        return self.df[k]

    def unique(self, strict=True):
        """
        Returns a dataframe with unique rows based on the set of labels

        :param strict: If true, limits to a stricter set by preferring those with non-`NA` values
        """
        d_set = dict(
            daeVt=['wall', 'origin', 'site'],
            daeVt_strict=['wall', 'origin']
        )
        col_nms = d_set[self.dnm]
        df = self.df.drop_duplicates(subset=col_nms).copy(deep=True)
        ic(df)
        ic(df['site'].cat.categories)
        if strict:
            col_nms_ = d_set[self.dnm+'_strict']
            ic(col_nms)

            df.sort_values(by=col_nms, inplace=True)
            # df.sort_values(by='site', inplace=True)
            # ic(df)
            # for col_nm in d_set[self.dnm+'_strict']:
            #     for c in self.df[col_nm].unique():
            #         if c != 'NA':
            #             ic(c)

        return df


if __name__ == '__main__':
    import os

    os.chdir('../PVC_DATA')

    el = EcgLabel('daeVt')
    ic(el.df)
    ic(el.df.dtypes)
    # ic(el.unique(strict=False))
    ic(el.unique())
    # ic(el['pat_num'])

