import numpy as np
import pandas as pd


def get_data(var, a=1.4, b=0.3, initial_values = [1, 1], iterations=1000):
    return get_dataframe(a,b, initial_values, iterations)[var].values


def get_dataframe(a=1.4, b=0.3, initial_values = [1, 1], iterations=1000):
    '''
    M. Hénon. "A two-dimensional mapping with a strange attractor". Commun. Math. Phys. 50, 69-77 (1976)

    dx/dt = a + by(t-1) - x(t-1)^2
    dy/dt = x

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