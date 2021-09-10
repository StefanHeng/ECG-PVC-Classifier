# """ **Obsolete**
# Export original MAT files to HDF5 format, with my custom formatting
# """
#
# import os
# from scipy.io import loadmat
# import h5py
#
# from icecream import ic
#
# os.chdir('../PVC_DATA')
#
#
# class EcgExport:
#     D = dict(
#         dae=dict(
#             path='MM_classifier/vtdata.mat',
#             keys=dict(
#                 vt_ecg='ecg',
#                 vtname='nm',
#                 vtcasename='cnm',
#                 vt_qrswidth='qrs_width'
#             )
#         ),
#     )
#
#     def __init__(self, d_nm):
#         d = self.D[d_nm]
#         mat = loadmat(d['path'])
#         fl_nm = f'Processed/{d_nm}.hdf5'
#         open(fl_nm, 'a').close()  # Create file in OS
#         f = h5py.File(fl_nm, 'w')
#
#         keys = d['keys']
#         # keys = filter(lambda k: k[:2] != '__' and k[-2:] != '__', mat.keys())
#         for k in keys:
#             data = mat[k]
#             if 1 in data.shape:
#                 data = data.reshape(data.size)
#             ic(k, data[:10])
#             # f.create_dataset(keys[k], data=data)
#
#         ic(f'Keys {keys} added for dataset {d_nm}')
#
#
# if __name__ == '__main__':
#     EcgExport('dae')
#
#     # extr = VibExtractFemto(num_spl, spl_rt)
#     # fl_nm = f'data/{fl_nm}.hdf5'
#     # open(fl_nm, 'a').close()  # Create file in OS
#     # fl = h5py.File(fl_nm, 'w')
#     # fl.attrs['feat_stor_idxs'] = json.dumps(self.ENC_FEAT_STOR)
#     # fl.attrs['brg_nms'] = json.dumps(self.FLDR_NMS)
#     # fl.attrs['nums_msr'] = json.dumps(self.NUMS_MESR)
#     # fl.attrs['feat_disp_nms'] = json.dumps({idx: nm for idx, nm in enumerate(extr.D_PROP_FUNC)})
#     # print(f'Metadata attributes created: {list(fl.attrs.keys())}')
#     # for idx_brg, test_nm in enumerate(self.FLDR_NMS):
#     #     group = fl.create_group(test_nm)
#     #     for acc in ['hori', 'vert']:
#     #         arr_extr = np.stack([
#     #             self.get_feature_series(idx_brg, func, acc) for k, func in extr.D_PROP_FUNC.items()
#     #         ])
#     #         group.create_dataset(acc, data=arr_extr)
