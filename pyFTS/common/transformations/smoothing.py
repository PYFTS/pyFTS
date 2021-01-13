from pyFTS.common.transformations.transformation import Transformation 
import numpy as np


class MovingAverage(Transformation):
    def __init__(self, **kwargs):
        super(MovingAverage, self).__init__()
        self.name = 'Moving Average Smoothing'
        self.steps = kwargs.get('steps',2)

    def apply(self, data, param=None, **kwargs):
        steps = param if param is not None else self.steps
        ma = [k for k in data[:steps]]
        ma.extend([np.mean(data[k-steps:k]) for k in range(steps, len(data))])
        return ma

    def inverse(self, data, param=None, **kwargs):
        return data


class ExponentialSmoothing(Transformation):
    def __init__(self, **kwargs):
        super(MovingAverage, self).__init__()
        self.name = 'Moving Average Smoothing'
        self.steps = kwargs.get('steps',2)
        self.beta = kwargs.get('beta',.5)

    def apply(self, data, param=None, **kwargs):
        steps = param if param is not None else self.steps
        beta = kwargs.get('beta',None)
        beta = beta if beta is not None else self.beta
        mm = [k for k in data[:steps]]
        for i in range(steps, len(data)):
            ret = 0
            for k in np.arange(0,steps):
                ret += ( beta * (1 - beta) ** k ) * data[i - k]
            mm.append(ret)
        return mm

    def inverse(self, data, param=None, **kwargs):
        return data
