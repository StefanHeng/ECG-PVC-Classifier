import numpy as np

import os

from icecream import ic

from util import *
from ecg_data import EcgData


os.chdir('../../PVC_DATA')


if __name__ == '__main__':
    dg = EcgData()
    ecgs = dg('daePm.ecg')
    ecgs_norm = normalize_signal(ecgs)

    sig_ori = ecgs[0]
    sig = ecgs_norm[0]
    ic(sig_ori, sig)

    plot_ecg_img(sig, save=True)

