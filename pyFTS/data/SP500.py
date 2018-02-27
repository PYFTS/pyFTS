import pandas as pd
import numpy as np
import os
import pkg_resources


def get_data():
    filename = pkg_resources.resource_filename('pyFTS', 'data/SP500.csv')
    dat = pd.read_csv(filename, sep=",")
    dat = np.array(dat["Avg"])
    return dat
