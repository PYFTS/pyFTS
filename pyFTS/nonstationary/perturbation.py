"""
Pertubation functions for Non Stationary Fuzzy Sets
"""

import numpy as np
from pyFTS import *
from pyFTS.common import FuzzySet, Membership


def linear(x, parameters):
    return np.polyval(parameters, x)


def polynomial(x, parameters):
        return np.polyval(parameters, x)


def exponential(x, parameters):
    return np.exp(x*parameters[0])


def periodic(x, parameters):
    return parameters[0] * np.sin(x * parameters[1])