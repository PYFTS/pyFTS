"""
FOREX market EUR-GBP pair.

Daily averaged quotations, by business day, from 2016 to 2018.
"""


from pyFTS.data import common
import pandas as pd
import numpy as np


def get_data():
    """
    Get the univariate time series data.

    :return: numpy array
    """
    dat = get_dataframe()
    return np.array(dat["Avg"])


def get_dataframe():
    """
    Get the complete multivariate time series data.

    :return: Pandas DataFrame
    """
    df = pd.read_csv('https://query.data.world/s/gvsaeruthnxjkwzl7z4ki7u5rduah3')

    return df

