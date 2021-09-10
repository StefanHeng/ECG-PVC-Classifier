import os
import scipy.io

from icecream import ic


if __name__ == '__main__':
    os.chdir('../PVC_DATA')

    mat = scipy.io.loadmat('MM_classifier/alldata.mat')
    ic(mat, mat.keys())

    casename = mat['vtcasename']  # An ndarray
    ic(casename, type(casename), len(casename))
    arr = casename[0]  # An ndarray
    ic(arr, type(arr), len(arr))

    arr = arr[0]  # An ndarray
    arr = arr[0]  # An ndarray
    ic(arr, type(arr), len(arr))
