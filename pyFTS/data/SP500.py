"""
S&P500 - Standard & Poor's 500

Daily averaged index, by business day, from 1950 to 2017.

Source: https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC
"""

from pyFTS.data import common
import pandas as pd
import numpy as np


def get_data() -> np.ndarray:
    """
    Get the univariate time series data.

    :return: numpy array
    """
    dat = get_dataframe()
    return np.array(dat["Avg"])


def get_dataframe() -> pd.DataFrame:
    """
    Get the complete multivariate time series data.

    :return: Pandas DataFrame
    """
    dat = common.get_dataframe('SP500.csv.bz2',
                               'https://github.com/petroniocandido/pyFTS/raw/8f20f3634aa6a8f58083bdcd1bbf93795e6ed767/pyFTS/data/SP500.csv.bz2',
                               sep=",", compression='bz2')
    return dat

