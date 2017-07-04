import numpy as np
import math
from pyFTS import *


class Transformation(object):
    """
    Data transformation used to pre and post processing of the FTS
    """

    def __init__(self, parameters):
        self.isInversible = True
        self.parameters = parameters
        self.minimalLength = 1

    def apply(self,data,param,**kwargs):
        pass

    def inverse(self,data, param,**kwargs):
        pass

    def __str__(self):
        return self.__class__.__name__ + '(' + str(self.parameters) + ')'


class Differential(Transformation):
    """
    Differentiation data transform
    """
    def __init__(self, parameters):
        super(Differential, self).__init__(parameters)
        self.lag = parameters
        self.minimalLength = 2

    def apply(self, data, param=None,**kwargs):
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

    def inverse(self,data, param, **kwargs):

        interval = kwargs.get("point_to_interval",False)

        if isinstance(data, (np.ndarray, np.generic)):
            data = data.tolist()

        if not isinstance(data, list):
            data = [data]

        if not isinstance(param, list):
            param = [param]

        n = len(data)

        if not interval:
            inc = [data[t] + param[t] for t in np.arange(0, n)]
        else:
            inc = [[data[t][0] + param[t], data[t][1] + param[t]] for t in np.arange(0, n)]

        if n == 1:
            return inc[0]
        else:
            return inc


class AdaptiveExpectation(Transformation):
    """
    Adaptive Expectation post processing
    """
    def __init__(self, parameters):
        super(AdaptiveExpectation, self).__init__(parameters)
        self.h = parameters

    def apply(self, data, param=None,**kwargs):
        return  data

    def inverse(self, data, param,**kwargs):
        n = len(data)

        inc = [param[t] + self.h*(data[t] - param[t]) for t in np.arange(0, n)]

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
