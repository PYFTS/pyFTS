import numpy as np
import math
from pyFTS import *


class Transformation(object):

    def __init__(self, parameters):
        self.isInversible = True
        self.parameters = parameters
        self.minimalLength = 1

    def apply(self,data,param):
        pass

    def inverse(self,data, param):
        pass

    def __str__(self):
        return self.__class__.__name__ + '(' + str(self.parameters) + ')'


class Differential(Transformation):

    def __init__(self, parameters):
        super(Differential, self).__init__(parameters)
        self.lag = parameters
        self.minimalLength = 2

    def apply(self, data, param=None):
        if param is not None:
            self.lag = param

        if not isinstance(data, (list, np.ndarray, np.generic)):
            data = [data]

        if isinstance(data, (np.ndarray, np.generic)):
            data = data.tolist()

        n = len(data)
        diff = [data[t - self.lag] - data[t] for t in np.arange(self.lag, n)]
        for t in np.arange(0, self.lag): diff.insert(0, 0)
        return diff

    def inverse(self,data, param):

        if isinstance(data, (np.ndarray, np.generic)):
            data = data.tolist()

        if not isinstance(data, list):
            data = [data]

        n = len(data)

        inc = [data[t] + param[t] for t in np.arange(0, n)]

        if n == 1:
            return inc[0]
        else:
            return inc


def boxcox(original, plambda):
    n = len(original)
    if plambda != 0:
        modified = [(original[t] ** plambda - 1) / plambda for t in np.arange(0, n)]
    else:
        modified = [math.log(original[t]) for t in np.arange(0, n)]
    return np.array(modified)


def Z(original):
    mu = np.mean(original)
    sigma = np.std(original)
    z = [(k - mu)/sigma for k in original]
    return z


# retrieved from Sadaei and Lee (2014) - Multilayer Stock ForecastingModel Using Fuzzy Time Series
def roi(original):
    n = len(original)
    roi = []
    for t in np.arange(0, n-1):
        roi.append( (original[t+1] - original[t])/original[t]  )
    return roi

def smoothing(original, lags):
    pass

def aggregate(original, operation):
    pass
