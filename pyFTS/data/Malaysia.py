"""
Hourly Malaysia eletric load and tempeature
"""


from pyFTS.data import common
import pandas as pd
import numpy as np


def get_data(field='load'):
    """
    Get the univariate time series data.

    :param field: dataset field to load
    :return: numpy array
    """
    dat = get_dataframe()
    return np.array(dat[field])


def get_dataframe():
    """
    Get the complete multivariate time series data.

    :return: Pandas DataFrame
    """
    df = common.get_dataframe("malaysia.csv","https://query.data.world/s/e5arbthdytod3m7wfcg7gmtluh3wa5",
                               sep=";")

    return df


