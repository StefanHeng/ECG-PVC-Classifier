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
    LB_TP = dict(  # Label types
        daeVt=['wall', 'side', 'ventricle']
    )
    STRICTS = dict(
        daeVt=['wall', 'side']  # Proper subset
    )

    def __init__(self, dnm: str):
        self.dnm = dnm
        self.label_types = self.LB_TP[dnm]

        rec = h5py.File(self.F, 'r')[dnm]
        columns = rec['columns'][:].astype('str')  # Per `EcgLabelExport`, bytes to strings
        dtypes = rec['dtypes'][:].astype('str')
        d = rec['data'][:].astype('str')
        self.df: pd.DataFrame = pd.DataFrame(d, columns=columns)
        for ds in dtypes:
            self.df = self.df.astype(dict(zip(columns, ds)))

        for idx, dtp in enumerate(dtypes[-1]):  # `NP` as the smallest category
            if dtp == 'category' and dtypes[0][idx] == 'string':
                col_nm = columns[idx]
                cat = self.df[col_nm].cat
                c_ = list(cat.categories)
                if 'NA' in c_:
                    idx = c_.index('NA')
                    c_[0], c_[idx] = c_[idx], c_[0]
                    self.df[col_nm] = cat.reorder_categories(c_)

    def __getitem__(self, k):
        return self.df[k]

    def unique(self, strict=True):
        """
        Returns a dataframe with unique rows based on the set of labels

        :param strict: If true, limits to a stricter set by preferring those with non-`NA` values
        """
        col_nms = self.LB_TP[self.dnm]
        df: pd.DataFrame = self.df.drop_duplicates(subset=col_nms).copy(deep=True)
        df.sort_values(by=col_nms, inplace=True)
        if strict:
            col_nms_ = self.STRICTS[self.dnm]

            # No values of `NA` is allowed for strict columns
            def no_na(c_nms):
                return lambda x_: all(x_[col_nm] != 'NA' for col_nm in c_nms)
            df = df[df.apply(no_na(col_nms_), axis=1)]

            df_exp = pd.DataFrame()

            # Keep only meaningful labels in non-strict columns if available
            for _, row in df.drop_duplicates(subset=col_nms_).iterrows():
                pairs = list(filter(lambda i: i[0] in col_nms_, row.items()))
                cols, vals = zip(*pairs)

                def same_vals(x_):
                    return all(x_[col_nm] == vals[idx] for idx, col_nm in enumerate(cols))

                df_i = df[df.apply(same_vals, axis=1)]

                # Only keep all non-NA rows for non-strict columns
                df_f = df_i[df_i.apply(no_na(list(set(col_nms) - set(col_nms_))), axis=1)]
                df_exp = df_exp.append(df_i if df_f.empty else df_f)
            df = df_exp

        return df


if __name__ == '__main__':
    import os

    os.chdir('../PVC_DATA')

    el = EcgLabel('daeVt')
    ic(el.df)
    ic(el.df.dtypes)
    ic(el.unique(strict=False))
    ic(el.unique())
    # ic(el['pat_num'])

