from pyFTS.common.transformations.transformation import Transformation 
import numpy as np

class MovingAverage(Transformation):
    def __init__(self, **kwargs):
        super(MovingAverage, self).__init__(**kwargs)
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
        super(ExponentialSmoothing,self).__init__(**kwargs)
        self.name = 'Exponential Moving Average Smoothing'
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


class AveragePooling(Transformation):
    def __init__(self, **kwargs):
        super(AveragePooling,self).__init__(**kwargs)
        self.name = 'Exponential Average Smoothing'
        self.kernel = kwargs.get('kernel',5)
        self.stride = kwargs.get('stride',1)
        self.padding = kwargs.get('padding','same')

    def apply(self, data):
        result = []
        if self.padding == 'same':
            for i in range(int(self.kernel/2), len(data)+int(self.kernel/2), self.stride):
                result.append(np.mean(data[np.max([0,i-self.kernel]):np.min([i, len(data)])]))

        elif self.padding == 'valid':
            for i in range(self.kernel, len(data), self.stride):
                result.append(np.mean(data[i-self.kernel:i]))
        else:
            raise ValueError('Invalid padding schema')
        return result

    def inverse(self, data, param=None, **kwargs):
        return data


class MaxPooling(Transformation):
    def __init__(self, **kwargs):
        super(MaxPooling,self).__init__(**kwargs)
        self.name = 'Exponential Average Smoothing'
        self.kernel = kwargs.get('kernel',5)
        self.stride = kwargs.get('stride',1)
        self.padding = kwargs.get('padding','same')

    def apply(self, data):
        result = []
        if self.padding == 'same':
            for i in range(int(self.kernel/2), len(data)+int(self.kernel/2), self.stride):
                result.append(np.max(data[np.max([0,i-self.kernel]):np.min([i, len(data)])]))

        elif self.padding == 'valid':
            for i in range(self.kernel - 1, len(data), self.stride):
                result.append(np.max(data[i-self.kernel:i]))
        else:
            raise ValueError('Invalid padding schema')
        return result

    def inverse(self, data, param=None, **kwargs):
        return data