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
            ic(df)
            ic(df.dtypes)
            ic(df.columns)

            columns = np.array(['pat_num', 'case_name', 'wall', 'origin', 'site'], dtype='S')
            dtypes = np.array(['int', 'S', 'S', 'S', 'S'], dtype='S')

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
                # pat_num = x['Name']
                # case_name = x['VT']
                # w_ = x['septum free wall']
                wall = d_wall[w] if w in d_wall else 'NA'
                # o_ = x['Intr']
                origin = d_origin[o] if o in d_origin else 'NA'
                if type(site) is float and isnan(site):
                    site = 'NA'
                # site = x['Unnamed: 6']
                # ic(pat_num, case_name, wall, origin, site)
                return pat_num, case_name, wall, origin, site

            df_ = df.apply(_map, axis=1, result_type='expand')
            return dict(
                columns=columns,
                dtypes=dtypes,
                data=df_.to_numpy().astype('S')
            )

        ks = ['daeVt']
        d_func = dict(
            daeVt=_daeVt
        )
        # for idx, k in enumerate(ks):
        #     d_func[k]().to_hdf(self.F, key=k, mode='w' if idx == 0 else 'a')

        open(self.F, 'a').close()  # Create file in OS
        f = h5py.File(self.F, 'w')
        for dnm in self.D.keys():
            group = f.create_group(dnm)
            for k, v in d_func[dnm]().items():
                group[k] = v


        # fl.attrs['feat_stor_idxs'] = json.dumps(self.ENC_FEAT_STOR)
        # fl.attrs['brg_nms'] = json.dumps(self.FLDR_NMS)
        # fl.attrs['nums_msr'] = json.dumps(self.NUMS_MESR)
        # fl.attrs['feat_disp_nms'] = json.dumps({idx: nm for idx, nm in enumerate(extr.D_PROP_FUNC)})
        # print(f'Metadata attributes created: {list(fl.attrs.keys())}')
        # for idx_brg, test_nm in enumerate(self.FLDR_NMS):
        #     group = fl.create_group(test_nm)
        #     for acc in ['hori', 'vert']:
        #         arr_extr = np.stack([
        #             self.get_feature_series(idx_brg, func, acc) for k, func in extr.D_PROP_FUNC.items()
        #         ])
        #         group.create_dataset(acc, data=arr_extr)
        # print(f'Features extracted: {[nm for nm in fl]}')


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

    # el = EcgLabelExport()
    record = h5py.File('ecg_label.hdf5', 'r')
    pp(record)

