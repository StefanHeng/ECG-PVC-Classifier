"""
Handles exporting ECG signal labels to consistent format
"""


import numpy as np
import pandas as pd
from math import isnan
from icecream import ic
import h5py


class EcgLabelExport:
    """
    Manages the labels corresponding to ECG signals as in EcgData
    """
    D = dict(  # Labels
        daeVt=dict(
            path='Data for Dae and Weiqing_myver.csv'
        )
    )
    F = 'ecg_label.hdf5'

    def __init__(self):
        def _daeVt():
            df = pd.read_csv(self.D['daeVt']['path'])

            # Actual numpy strings with `U` not compatible with `hdf5`
            dtype_exp = 'S'
            columns = np.array(['pat_num', 'case_name', 'wall', 'origin', 'site'], dtype=dtype_exp)
            dtypes = np.array([
                ['int', 'string', 'string', 'string', 'string'],  # 1st run
                ['category', 'category', 'category', 'category', 'category']  # 2nd run
            ], dtype=dtype_exp)

            d_wall = {
                'free wall': 'FW',
                'septum': 'SP'
            }
            d_origin = dict(
                epi='EP',
                endo='ED',
                intramural='IM'
            )

            def _map(x):
                pat_num, case_name, w, _, o, __, site = x
                wall = d_wall[w] if w in d_wall else 'NA'
                origin = d_origin[o] if o in d_origin else 'NA'
                if type(site) is float and isnan(site):
                    site = 'NA'
                return pat_num, case_name, wall, origin, site

            df_ = df.apply(_map, axis=1, result_type='expand')
            return dict(
                columns=columns,
                dtypes=dtypes,
                data=df_.to_numpy().astype(dtype_exp)
            )

        d_func = dict(
            daeVt=_daeVt
        )

        open(self.F, 'a').close()  # Create file in OS
        f = h5py.File(self.F, 'w')
        for dnm in self.D.keys():
            group = f.create_group(dnm)
            d = d_func[dnm]()
            for k, v in d.items():
                group[k] = v
            print(f'Dataset {dnm} labels generated with keys {list(d.keys())}')

        print(f'Dataset labels generated for {list(self.D.keys())}')


def pp(rec):
    """ Pretty print hdf5 file content """
    for k in rec.keys():
        grp = rec[k]
        ic(f'Group {grp}')
        for k_ in grp.keys():
            d = grp[k_]
            ic(d, d[:5])


if __name__ == '__main__':
    import os
    os.chdir('../PVC_DATA')

    EcgLabelExport()
    record = h5py.File('ecg_label.hdf5', 'r')
    pp(record)

