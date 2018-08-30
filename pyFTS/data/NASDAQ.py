"""
National Association of Securities Dealers Automated Quotations - Composite Index (NASDAQ IXIC)

Daily averaged index by business day, from 2000 to 2016.

Source: http://www.nasdaq.com/aspx/flashquotes.aspx?symbol=IXIC&selected=IXIC
"""

from pyFTS.data import common
import pandas as pd
import numpy as np


def get_data(field="avg"):
    """
    Get a simple univariate time series data.

    :param field: the dataset field name to extract
    :return: numpy array
    """
    dat = get_dataframe()
    dat = np.array(dat[field])
    return dat


def get_dataframe():
    """
    Get the complete multivariate time series data.

    :return: Pandas DataFrame
    """
    dat = common.get_dataframe('NASDAQ.csv.bz2',
                               'https://github.com/petroniocandido/pyFTS/raw/8f20f3634aa6a8f58083bdcd1bbf93795e6ed767/pyFTS/data/NASDAQ.csv.bz2',
                               sep=",", compression='bz2')
    return dat

