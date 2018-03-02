import pandas as pd
import numpy as np
import os
import pkg_resources


def get_data():
    filename = pkg_resources.resource_filename('pyFTS', 'data/SP500.csv.bz2')
    dat = pd.read_csv(filename, sep=",", compression='bz2')
    dat = np.array(dat["Avg"])
    return dat
