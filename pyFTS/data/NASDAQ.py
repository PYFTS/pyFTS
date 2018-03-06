from pyFTS.data import common
import pandas as pd
import numpy as np


def get_data(field):
    dat = get_dataframe()
    dat = np.array(dat[field])
    return dat


def get_dataframe():
    dat = common.get_dataframe('NASDAQ.csv.bz2',
                               'https://github.com/petroniocandido/pyFTS/raw/8f20f3634aa6a8f58083bdcd1bbf93795e6ed767/pyFTS/data/NASDAQ.csv.bz2',
                               sep=";", compression='bz2')
    return dat

