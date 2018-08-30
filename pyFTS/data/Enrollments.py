"""
Yearly University of Alabama enrollments from 1971 to 1992.
"""

from pyFTS.data import common
import pandas as pd
import numpy as np


def get_data():
    """
    Get a simple univariate time series data.

    :return: numpy array
    """
    dat = get_dataframe()
    dat = np.array(dat["Enrollments"])
    return dat


def get_dataframe():
    dat = common.get_dataframe('Enrollments.csv',
                               'https://github.com/petroniocandido/pyFTS/raw/8f20f3634aa6a8f58083bdcd1bbf93795e6ed767/pyFTS/data/Enrollments.csv',
                               sep=";")
    return dat
