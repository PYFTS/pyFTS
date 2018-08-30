"""
Lorenz, Edward Norton (1963). "Deterministic nonperiodic flow". Journal of the Atmospheric Sciences. 20 (2): 130–141.
https://doi.org/10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2

dx/dt = a(y -x)
dy/dt = x(b - z) - y
dz/dt = xy - cz
"""

import numpy as np
import pandas as pd


def get_data(var, a = 10.0, b = 28.0, c = 8.0 / 3.0, dt = 0.01,
                  initial_values = [0.1, 0, 0], iterations=1000):
    """
        Get a simple univariate time series data.

        :param var: the dataset field name to extract
        :return: numpy array
        """
    return get_dataframe(a, b, c, dt, initial_values, iterations)[var].values


def get_dataframe(a = 10.0, b = 28.0, c = 8.0 / 3.0, dt = 0.01,
                  initial_values = [0.1, 0, 0], iterations=1000):
    '''
    Return a dataframe with the multivariate Lorenz Map time series (x, y, z).

    :param a: Equation coefficient. Default value: 10
    :param b: Equation coefficient. Default value: 28
    :param c: Equation coefficient. Default value: 8.0/3.0
    :param dt: Time differential for continuous time integration. Default value: 0.01
    :param initial_values: numpy array with the initial values of x,y and z. Default: [0.1, 0, 0]
    :param iterations: number of iterations. Default: 1000
    :return: Panda dataframe with the x, y and z values
    '''

    x = [initial_values[0]]
    y = [initial_values[1]]
    z = [initial_values[2]]

    for t in np.arange(0, iterations):
        dxdt = a * (y[t] - x[t])
        dydt = x[t] * (b - z[t]) - y[t]
        dzdt = x[t] * y[t] - c * z[t]
        x.append(x[t] + dt * dxdt)
        y.append(y[t] + dt * dydt)
        z.append(z[t] + dt * dzdt)

    return pd.DataFrame({'x': x, 'y':y, 'z': z})