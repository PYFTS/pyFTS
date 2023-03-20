"""
DJI - Dow Jones

Daily averaged index, by business day, from 1985 to 2017.

Source: https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC
"""


from pyFTS.data import common
import pandas as pd
import numpy as np


def get_data(field:str='AVG') -> np.ndarray:
    """
    Get the univariate time series data.

    :param field: dataset field to load
    :return: numpy array
    """
    dat = get_dataframe()
    return np.array(dat[field])


def get_dataframe() -> pd.DataFrame:
    """
    Get the complete multivariate time series data.

    :return: Pandas DataFrame
    """
    df = common.get_dataframe("DowJones.csv", "https://query.data.world/s/d4hfir3xrelkx33o3bfs5dbhyiztml",
                              sep=",")

    return df

