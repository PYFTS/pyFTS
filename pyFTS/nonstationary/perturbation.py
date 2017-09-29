"""
Pertubation functions for Non Stationary Fuzzy Sets
"""

import numpy as np
from pyFTS import *
from pyFTS.common import FuzzySet, Membership


def linear(x, parameters):
    return parameters[0]*x + parameters[1]


def polynomial(x, parameters):
    n = len(parameters)
    tmp = 0.0
    for k in np.arange(0,n):
        tmp += parameters[k] * x**k
    return tmp


def exponential(x, parameters):
    return np.exp(x*parameters[0])


def periodic(x, parameters):
    return np.sin(x * parameters[0] + parameters[1])