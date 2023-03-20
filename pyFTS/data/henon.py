"""
M. Hénon. "A two-dimensional mapping with a strange attractor". Commun. Math. Phys. 50, 69-77 (1976)

dx/dt = a + by(t-1) - x(t-1)^2
dy/dt = x
"""

import numpy as np
import pandas as pd


def get_data(var: str, a:float=1.4, b:float=0.3, initial_values: list = [1, 1], iterations:int=1000) -> pd.DataFrame:
    """
    Get a simple univariate time series data.

    :param var: the dataset field name to extract
    :return: numpy array
    """
    return get_dataframe(a,b, initial_values, iterations)[var].values


def get_dataframe(a:float=1.4, b:float=0.3, initial_values: list = [1, 1], iterations:int=1000) -> pd.DataFrame:
    '''
    Return a dataframe with the bivariate Henon Map time series (x, y).

    :param a: Equation coefficient
    :param b: Equation coefficient
    :param initial_values: numpy array with the initial values of x and y. Default: [1, 1]
    :param iterations: number of iterations. Default: 1000
    :return: Panda dataframe with the x and y values
    '''

    x = [initial_values[0]]
    y = [initial_values[1]]
    for t in np.arange(0, iterations):
        xx = a + b * y[t] - x[t] ** 2
        y.append(x[t])
        x.append(xx)

    return pd.DataFrame({'x': x, 'y':y})