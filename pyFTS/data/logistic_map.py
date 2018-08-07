#

import numpy as np


def get_data(r = 4, initial_value = 0.3, iterations=100):
    '''
    May, Robert M. (1976). "Simple mathematical models with very complicated dynamics".
    Nature. 261 (5560): 459–467. doi:10.1038/261459a0.

    x(t) = r * x(t-1) * (1 - x(t -1) )

    :param r: Equation coefficient
    :param initial_value: Initial value of x. Default: 0.3
    :param iterations: number of iterations. Default: 100
    :return:
    '''
    x = [initial_value]
    for t in np.arange(0,iterations):
      x.append(r * x[t]*(1 - x[t]))

    return x