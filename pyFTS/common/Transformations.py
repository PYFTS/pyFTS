import numpy as np
import math
from pyFTS import *


class Transformation(object):
    """
    Data transformation used to pre and post processing of the FTS
    """

    def __init__(self, **kwargs):
        self.is_invertible = True
        self.minimal_length = 1

    def apply(self, data, param, **kwargs):
        pass

    def inverse(self,data, param, **kwargs):
        pass

    def __str__(self):
        return self.__class__.__name__ + '(' + str(self.parameters) + ')'


class Differential(Transformation):
    """
    Differentiation data transform
    """
    def __init__(self, parameters):
        super(Differential, self).__init__()
        self.lag = parameters
        self.minimal_length = 2

    def apply(self, data, param=None, **kwargs):
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

    def inverse(self, data, param, **kwargs):

        interval = kwargs.get("point_to_interval",False)

        if isinstance(data, (np.ndarray, np.generic)):
            data = data.tolist()

        if not isinstance(data, list):
            data = [data]

        n = len(data)

#        print(n)
#        print(len(param))

        if not interval:
            inc = [data[t] + param[t] for t in np.arange(0, n)]
        else:
            inc = [[data[t][0] + param[t], data[t][1] + param[t]] for t in np.arange(0, n)]

        if n == 1:
            return inc[0]
        else:
            return inc


class Scale(Transformation):
    def __init__(self, min=0, max=1):
        super(Scale, self).__init__()
        self.data_max = None
        self.data_min = None
        self.transf_max = max
        self.transf_min = min

    def apply(self, data, param=None,**kwargs):
        if self.data_max is None:
            self.data_max = np.nanmax(data)
            self.data_min = np.nanmin(data)
        data_range = self.data_max - self.data_min
        transf_range = self.transf_max - self.transf_min
        if isinstance(data, list):
            tmp = [(k + (-1 * self.data_min)) / data_range for k in data]
            tmp2 = [ (k * transf_range) + self.transf_min for k in tmp]
        else:
            tmp = (data + (-1 * self.data_min)) / data_range
            tmp2 = (tmp * transf_range) + self.transf_min

        return  tmp2

    def inverse(self, data, param, **kwargs):
        data_range = self.data_max - self.data_min
        transf_range = self.transf_max - self.transf_min
        if isinstance(data, list):
            tmp2 = [(k - self.transf_min) / transf_range   for k in data]
            tmp = [(k * data_range) + self.data_min for k in tmp2]
        else:
            tmp2 = (data - self.transf_min) / transf_range
            tmp = (tmp2 * data_range) + self.data_min
        return tmp


class AdaptiveExpectation(Transformation):
    """
    Adaptive Expectation post processing
    """
    def __init__(self, parameters):
        super(AdaptiveExpectation, self).__init__(parameters)
        self.h = parameters

    def apply(self, data, param=None,**kwargs):
        return data

    def inverse(self, data, param,**kwargs):
        n = len(data)

        inc = [param[t] + self.h*(data[t] - param[t]) for t in np.arange(0, n)]

        if n == 1:
            return inc[0]
        else:
            return inc


class BoxCox(Transformation):
    def __init__(self, plambda):
        super(BoxCox, self).__init__()
        self.plambda = plambda

    def apply(self, data, param=None, **kwargs):
        if self.plambda != 0:
            modified = [(dat ** self.plambda - 1) / self.plambda for dat in data]
        else:
            modified = [np.log(dat) for dat in data]
        return np.array(modified)

    def inverse(self, data, param=None, **kwargs):
        if self.plambda != 0:
            modified = [np.exp(np.log(dat * self.plambda + 1) ) / self.plambda for dat in data]
        else:
            modified = [np.exp(dat) for dat in data]
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
