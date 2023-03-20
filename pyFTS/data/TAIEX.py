"""
The Taiwan Stock Exchange Capitalization Weighted Stock Index (TAIEX)

Daily averaged index by business day, from 1995 to 2014.

Source: http://www.twse.com.tw/en/products/indices/Index_Series.php
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
    dat = np.array(dat["avg"])
    return dat


def get_dataframe() -> pd.DataFrame:
    """
    Get the complete multivariate time series data.

    :return: Pandas DataFrame
    """
    dat = common.get_dataframe('TAIEX.csv.bz2',
                               'https://github.com/petroniocandido/pyFTS/raw/8f20f3634aa6a8f58083bdcd1bbf93795e6ed767/pyFTS/data/TAIEX.csv.bz2',
                               sep=",", compression='bz2')
    dat["Date"] = pd.to_datetime(dat["Date"])
    return dat

