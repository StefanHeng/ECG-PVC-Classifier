import numpy as np

from scipy.io import loadmat

import re
from typing import Union

from icecream import ic


class DataGetter:
    """
    Interface to fetch ECG data from .MAT files in the datasets
    """
    class Dataset:
        """
        Interface to MAT file dictionary output
        """

        D = dict(
            daePm=dict(
                path='MM_classifier/alldata.mat',
                keys=dict(
                    ecg='pm_ecg',
                    origin='pm_epiendo',  # Binary flag, does 0 mean EPI or ENDO?
                    name='pmname',
                    case_name='pmcasename',
                    qrs_width='pm_qrswidth'
                )
            ),
            daePmName=dict(
                path='MM_classifier/allname.mat',
                keys=dict(
                    case_name='pmcasename'  # Seems redundant for already part of `daePm`
                )
            ),
            daePmFw=dict(
                path='MM_classifier/freewall.mat',
                keys=dict(
                    case_name='pmcasename1'  # TODO: Is this info somewhere else too?
                )
            ),
            daeVt=dict(
                path='MM_classifier/vtdata.mat',
                keys=dict(
                    ecg='vt_ecg',
                    name='vtname',
                    case_name='vtcasename',
                    qrs_width='vt_qrswidth'
                )
            ),
            daeRaw=dict(
                path='MM_classifier/ecg_nonnorm.mat',
                keys=dict(
                    pm='pm_ecg',
                    vt='vt_ecg',
                    pvc='pvc_ecg'
                )
            ),
            daeRawCnm=dict(
                path='MM_classifier/filename.mat',
                keys=dict(
                    vt='vtcasename',
                    pvc='pvccasename',
                    pm='pmcasename'
                )
            ),
            daePmCtrl=dict(
                path='MM_classifier/allcontrol.mat',  # TODO: What's this for?
                keys=dict(
                    ecg='pm_ecg',
                    origin='pm_epiendo',
                    qrs_width='pm_qrswidth',
                    name='pmname',
                    case_name='pmcasename'
                )
            ),
            daePvcCtrl=dict(
                path='MM_classifier/allcontrol.mat',
                keys=dict(
                    ecg='pvc_ecg',
                    qrs_width='pvc_qrswidth',
                    name='pvcname',
                    case_name='pvccasename'
                )
            ),
            daePmSlope=dict(
                path='MM_classifier/pm_slopes.mat',
                keys=dict(
                    slope='slope_res',
                    name='pmname',
                    case_name='pmcasename'
                )
            ),
            daeVtSlope=dict(
                path='MM_classifier/vtslope.mat',
                keys=dict(
                    slope='slope_res',
                    qrs_width='vt_qrswidth',
                    name='vtname',
                    case_name='vtcasename'
                )
            ),
            daeVtSlope32=dict(
                path='MM_classifier/vt_slopefea32_1234.mat',
                keys=dict(
                    lbbb='vt_lbbb',  # TODO: ??
                    qrs_width='vt_qrswidth',
                    pat_num='vt_patnum',
                    pace_num='vt_pacenum',
                    category='vt_category',
                    origin='vt_epiendo',
                    slope='vt_all'  # TODO: Just guessing, what's the feature?
                )
            ),
            daeSup=dict(
                path='MM_classifier/Superior.mat',  # Superior or inferior axis?
                keys=dict(
                    slope='slope_res1',
                    case_name='aa2'  # Why the original key name?
                )
            ),
            daeInf=dict(
                path='MM_classifier/inferior.mat',
                keys=dict(
                    slope='slope_res',
                    case_name='aa1'
                )
            ),
            daeSimul=dict(
                path='MM_classifier/simul_123456789.mat',  # TODO: What is this for?
                keys=dict(
                    patients='patient_list',  # Why is this here?
                    pat_num='pm_patnum',  # What is this? Patient number?
                    pace_num='pm_pacenum',
                    origin='pm_epiendo',
                    lbbb='pm_lbbb',  # ?
                    qrs_width='pm_qrswidth',
                    category='pm_category',  # ?
                    septum='pm_septum',
                    qrs_stim='pm_stimqrs',
                    volt='pm_volt',
                    all='pm_all'  # TODO: ?
                )
            ),
            daePmLst=dict(
                path='MM_classifier/pacemaplist_matlab_dyj.mat',
                keys=dict(
                    pat_num='pmlist_patnum',
                    pace_num='pmlist_pacenum',
                    pace_site='pmlist_pacesite',  # How is this encoded?
                    origin='pmlist_epiendo',
                    septum='pmlist_septum',
                    qrs_stim='pmlist_stimqrs',
                    volt='pmlist_volt',
                    date='pmlist_date'  # Encoding?
                )
            ),
            daeVtLst=dict(
                path='MM_classifier/vtlist_matlab_dyj.mat',
                keys=dict(
                    pat_num='vtlist_patnum',
                    pace_num='vtlist_pacenum',
                    pace_site='vtlist_pacesite',
                    pace_map='vtlist_pacemap',  # TODO: What is this, seems like the case name
                    origin='vtlist_epiendo'
                )
            ),
            daeLv=dict(
                path='MM_classifier/lvsam.mat',  # Left ventricle samples?
                keys=dict(
                    slope='slope_res1',
                    qrs_width='lvqrs',
                    name='lvna',
                    pat_num='lvpat'
                )
            ),
            daeRv=dict(
                path='MM_classifier/rvsam.mat',
                keys=dict(
                    slope='slope_res',
                    qrs_width='rvqrs',
                    name='rvna',
                    pat_num='rvpat'
                )
            )
        )

        def __init__(self, d_nm: str):
            """
            :param d_nm: Database code
            """
            self.nm = d_nm
            attrs = self.D[d_nm]
            self.path = attrs['path']
            self.key_map = attrs['keys']
            self.keys = list(self.key_map.keys())

            self.d = {}
            mat = loadmat(self.path)
            for k, v in self.key_map.items():
                self.d[k] = mat[v]

            self.meta = dict(
                dim_ecg=None,
                num_pat=None,
                n=None
            )

            for k, data in self.d.items():
                s = data.shape

                if 1 in s:
                    self.d[k] = data = data.reshape(tuple(filter(lambda x: x != 1, s)))
                    s = data.shape

                if len(s) == 2:  # Expect ecg signal
                    self.meta['dim_ecg'] = s[-1]

                self.meta['n'] = s[0]

                # Deals with how data is stored in the .mat files
                if data.dtype is np.dtype('O') and data[0].dtype.type is np.str_:
                    self.d[k] = data = np.vectorize(lambda x: x[0])(data)

                if data.dtype.type is np.str_ and data[0].endswith('.mat'):
                    self.d[k] = np.vectorize(lambda x: x.removesuffix('.mat'))(data)

            if d_nm == 'daeRaw':
                def w(x):
                    return f'{x}_width'

                def c(x):
                    return f'{x}_case_name'

                for k in self.keys:
                    self.d[w(k)] = np.count_nonzero(~np.isnan(self.d[k]), axis=1)

                self.keys += list(map(w, self.keys))

                case_names = DataGetter.Dataset(f'{d_nm}Cnm')
                keys = case_names.keys
                for k in keys:
                    self.d[c(k)] = case_names.d[k]
                self.keys += list(map(c, keys))

            def _get_num_patient():
                def _get(k='name'):
                    # return np.unique(dg(f'{d_nm}.{k}')).size
                    return np.unique(self.d[k]).size

                def _get_names(arr):
                    return np.vectorize(lambda x: re.match(r'([0-9]+)_', x).groups()[0])(arr)

                def _get_raw():
                    d = dict()
                    for key in ['vt', 'pvc', 'pm']:
                        key_nm = f'{key}_case_name'
                        d[key] = np.unique(_get_names(self.d[key_nm])).size
                    return d

                if d_nm in ['daeVt', 'daePm', 'daePmCtrl', 'daePvcCtrl', 'daePmSlope', 'daeVtSlope', 'daeLv', 'daeRv']:
                    return _get()
                elif d_nm in ['daeRaw']:
                    return _get_raw()
                elif d_nm in ['daeVtSlope32', 'daeSimul', 'daePmLst', 'daeVtLst']:
                    return _get(k='pat_num')
                elif d_nm in ['daeSup', 'daeInf', 'daePmName', 'daePmFw']:
                    return np.unique(_get_names(self.d['case_name'])).size
                else:
                    return None

            self.meta['num_pat'] = _get_num_patient()

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
                for key in self.keys:
                    d[key] = self.__getitem__(key)[k]
                for k, v in d.items():
                    if v.dtype.type is np.str_:
                        d[k] = v[0].removesuffix('.mat')
                return d

        def __str__(self):
            return self.d.__str__()

        def overview(self, n=30, show_ori=True):
            if show_ori:
                ic(self.nm, self.path)
            else:
                ic(self.nm)

            for k in self.keys:
                data = self.d[k]
                s = data.shape
                view = np.unique(data) if len(s) == 1 else data[0]  # Expect 2-dim
                view = view[:(int(n/3) if view[0].dtype.type is np.str_ else n)]
                if show_ori and k in self.key_map:
                    ic(k, self.key_map[k], s, view[:n])
                else:
                    ic(k, s, view[:n])
            ic('\n')  # built-in `print` isn't synchronized

    DSETS = list(Dataset.D.keys())

    def __call__(self, k: Union[str, list]):
        """
        :param k: Key, determines which data to fetch. Expect a dot(`.`)-separated string or a list

        The 1st string specifies which dataset, If only 1 code is provided, the entire dictionary is returned
        The 2nd string or integer, if provided, returns the corresponding key in the data set

        The hierarchy of valid codes: {
            - 'dae': Based on `Data for Dae and Weiqing_myver`, n=156
                - 'ecg': The ECG signal
                - 'qrs_width': The width of QRS complex
                - 'nm': Name
                - 'cNm': Case name
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

    import os
    os.chdir('../PVC_DATA')

    dg = DataGetter()

    idx = 20
    dataset = dg('daeVt')
    ecg = dg(['daeVt', 'ecg'])
    nm = dg('daeVt.name')
    cNm = dataset['case_name']
    width = dataset['qrs_width']

    for i in [ecg, nm, cNm, width]:
        ic(i[:10], i.shape)

    ds = [
        dataset[str(idx)],
        dataset[idx + 1],
        dg(['daeVt', idx + 2]),
        dg(['daeVt', f'{idx + 3}'])
    ]
    ic(ds[0])
    # for ecg_d in ds:
    #     arr = ecg_d['ecg']
    #     nm = ecg_d['cNm']
    #     plot_single(arr, nm)

    dataset = dg('daePm')
    ic(dataset.d)
