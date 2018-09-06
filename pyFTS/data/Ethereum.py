"""
Ethereum to USD quotations

Daily averaged index, by business day, from 2016 to 2018.

Source: https://finance.yahoo.com/quote/ETH-USD?p=ETH-USD
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
    df = pd.read_csv('https://query.data.world/s/qj4ly7o4rl7oq527xzy4v76wkr3hws')

    return df

