from pyFTS.data import common
import pandas as pd
import numpy as np


def get_dataframe():
    dat = common.get_dataframe('SP500.csv.bz2',
                               'https://github.com/petroniocandido/pyFTS/raw/8f20f3634aa6a8f58083bdcd1bbf93795e6ed767/pyFTS/data/SP500.csv.bz2',
                               sep=",", compression='bz2')
    dat = np.array(dat["Avg"])
    return dat

