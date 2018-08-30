"""
Monthly totals of a airline passengers from USA, from January 1949 through December 1960.

Source: Hyndman, R.J., Time Series Data Library, http://www-personal.buseco.monash.edu.au/~hyndman/TSDL/.
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
    dat = np.array(dat["Passengers"])
    return dat

def get_dataframe():
    """
    Get the complete multivariate time series data.

    :return: Pandas DataFrame
    """
    dat = common.get_dataframe('AirPassengers.csv',
                               'https://github.com/petroniocandido/pyFTS/raw/8f20f3634aa6a8f58083bdcd1bbf93795e6ed767/pyFTS/data/AirPassengers.csv',
                               sep=",")
    return dat

