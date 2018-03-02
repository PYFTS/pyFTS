import pandas as pd
import numpy as np
import os
import pkg_resources


def get_data():
    filename = pkg_resources.resource_filename('pyFTS', 'data/TAIEX.csv.bz2')
    dat = pd.read_csv(filename, sep=",", compression='bz2')
    dat = np.array(dat["avg"])
    return dat


def get_dataframe():
    filename = pkg_resources.resource_filename('pyFTS', 'data/TAIEX.csv.bz2')
    dat = pd.read_csv(filename, sep=",", compression='bz2')
    dat["Date"] = pd.to_datetime(dat["Date"])
    return dat
