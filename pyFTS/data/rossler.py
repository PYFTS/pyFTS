"""
O. E. Rössler, Phys. Lett. 57A, 397 (1976).

dx/dt = -z - y
dy/dt = x + ay
dz/dt = b + z( x - c )

"""

import numpy as np
import pandas as pd


def get_data(var, a = 0.2, b = 0.2, c = 5.7, dt = 0.01,
                  initial_values = [0.001, 0.001, 0.001], iterations=5000):
    """
    Get a simple univariate time series data.

    :param var: the dataset field name to extract
    :return: numpy array
    """
    return get_dataframe(a, b, c, dt, initial_values, iterations)[var].values


def get_dataframe(a = 0.2, b = 0.2, c = 5.7, dt = 0.01,
                  initial_values = [0.001, 0.001, 0.001], iterations=5000):
    '''
    Return a dataframe with the multivariate Rössler Map time series (x, y, z).

    :param a: Equation coefficient. Default value: 0.2
    :param b: Equation coefficient. Default value: 0.2
    :param c: Equation coefficient. Default value: 5.7
    :param dt: Time differential for continuous time integration. Default value: 0.01
    :param initial_values: numpy array with the initial values of x,y and z. Default: [0.001, 0.001, 0.001]
    :param iterations: number of iterations. Default: 5000
    :return: Panda dataframe with the x, y and z values
    '''

    x = [initial_values[0]]
    y = [initial_values[1]]
    z = [initial_values[2]]

    for t in np.arange(0, iterations):
        dxdt = - (y[t] + z[t])
        dydt = x[t] + a * y[t]
        dzdt = b + z[t] * x[t] - z[t] * c
        x.append(x[t] + dt * dxdt)
        y.append(y[t] + dt * dydt)
        z.append(z[t] + dt * dzdt)

    return pd.DataFrame({'x': x, 'y':y, 'z': z})