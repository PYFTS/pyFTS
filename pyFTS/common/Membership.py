"""
Membership functions for Fuzzy Sets
"""

import numpy as np
import math
from pyFTS import *


def trimf(x, parameters):
    """
    Triangular fuzzy membership function  
    :param x: data point
    :param parameters: a list with 3 real values
    :return: the membership value of x given the parameters
    """
    xx = np.round(x, 3)
    if xx < parameters[0]:
        return 0
    elif parameters[0] <= xx < parameters[1]:
        return (x - parameters[0]) / (parameters[1] - parameters[0])
    elif parameters[1] <= xx <= parameters[2]:
        return (parameters[2] - xx) / (parameters[2] - parameters[1])
    else:
        return 0


def trapmf(x, parameters):
    """
    Trapezoidal fuzzy membership function  
    :param x: data point
    :param parameters: a list with 4 real values
    :return: the membership value of x given the parameters
    """
    if x < parameters[0]:
        return 0
    elif parameters[0] <= x < parameters[1]:
        return (x - parameters[0]) / (parameters[1] - parameters[0])
    elif parameters[1] <= x <= parameters[2]:
        return 1
    elif parameters[2] <= x <= parameters[3]:
        return (parameters[3] - x) / (parameters[3] - parameters[2])
    else:
        return 0


def gaussmf(x, parameters):
    """
    Gaussian fuzzy membership function  
    :param x: data point
    :param parameters: a list with 2 real values (mean and variance)
    :return: the membership value of x given the parameters
    """
    return math.exp((-(x - parameters[0])**2)/(2 * parameters[1]**2))


def bellmf(x, parameters):
    """
    Bell shaped membership function
    :param x:
    :param parameters:
    :return:
    """
    return 1 / (1 + abs((x - parameters[2]) / parameters[0]) ** (2 * parameters[1]))


def sigmf(x, parameters):
    """
    Sigmoid / Logistic membership function
    :param x:
    :param parameters: an list with 2 real values (smoothness and midpoint)
    :return:
    """
    return 1 / (1 + math.exp(-parameters[0] * (x - parameters[1])))
