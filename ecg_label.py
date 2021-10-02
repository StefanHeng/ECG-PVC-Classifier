import pandas as pd
from icecream import ic


class EcgLabel:
    """
    Manages the labels corresponding to ECG signals as in EcgData
    """
    F = 'ecg_label.hdf5'

    def __init__(self, dnm: str):
        self.df = pd.read_hdf(self.F, dnm)


if __name__ == '__main__':
    el = EcgLabel('daeVt')

