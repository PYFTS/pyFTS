import pandas as pd
import numpy as np
import os
import pkg_resources


def get_data(field):
    filename = pkg_resources.resource_filename('pyFTS', 'data/SONDA_BSB.csv.bz2')
    dat = pd.read_csv(filename, sep=";", compression='bz2')
    dat = np.array(dat[field])
    return dat


def get_dataframe():
    filename = pkg_resources.resource_filename('pyFTS', 'data/SONDA_BSB.csv.bz2')
    dat = pd.read_csv(filename, sep=";", compression='bz2')
    dat["datahora"] = pd.to_datetime(dat["datahora"], format='%Y-%m-%d %H:%M:%S')
    return dat
