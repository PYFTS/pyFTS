from pyFTS.data import common
import pandas as pd
import numpy as np

def get_data():
    dat = get_dataframe()
    dat = np.array(dat["SUNACTIVITY"])
    return dat

def get_dataframe():
    dat = common.get_dataframe('sunspots.csv',
                               'https://github.com/petroniocandido/pyFTS/raw/8f20f3634aa6a8f58083bdcd1bbf93795e6ed767/pyFTS/data/sunspots.csv',
                               sep=",")
    return dat

