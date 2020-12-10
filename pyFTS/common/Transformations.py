"""
Common data transformation used on pre and post processing of the FTS
"""

import numpy as np
import pandas as pd
import math
from pyFTS import *


class Transformation(object):
    """
    Data transformation used on pre and post processing of the FTS
    """

    def __init__(self, **kwargs):
        self.is_invertible = True
        self.is_multivariate = False
        """detemine if this transformation can be applied to multivariate data"""
        self.minimal_length = 1
        self.name = ''

    def apply(self, data, param, **kwargs):
        """
        Apply the transformation on input data

        :param data: input data
        :param param:
        :param kwargs:
        :return: numpy array with transformed data
        """
        pass

    def inverse(self,data, param, **kwargs):
        """

        :param data: transformed data
        :param param:
        :param kwargs:
        :return: numpy array with inverse transformed data
        """
        pass

    def __str__(self):
        return self.name


class Differential(Transformation):
    """
    Differentiation data transform

    y'(t) = y(t) - y(t-1)
    y(t) =  y(t-1)  + y'(t)
    """
    def __init__(self, lag):
        super(Differential, self).__init__()
        self.lag = lag
        self.minimal_length = 2
        self.name = 'Diff'

    @property
    def parameters(self):
        return self.lag

    def apply(self, data, param=None, **kwargs):
        if param is not None:
            self.lag = param

        if not isinstance(data, (list, np.ndarray, np.generic)):
            data = [data]

        if isinstance(data, (np.ndarray, np.generic)):
            data = data.tolist()

        n = len(data)
        diff = [data[t] - data[t - self.lag] for t in np.arange(self.lag, n)]
        for t in np.arange(0, self.lag): diff.insert(0, 0)
        return diff

    def inverse(self, data, param, **kwargs):

        type = kwargs.get("type","point")
        steps_ahead = kwargs.get("steps_ahead", 1)

        if isinstance(data, (np.ndarray, np.generic)):
            data = data.tolist()

        if not isinstance(data, list):
            data = [data]

        n = len(data)

#        print(n)
#        print(len(param))

        if steps_ahead == 1:
            if type == "point":
                inc = [data[t] + param[t] for t in np.arange(0, n)]
            elif type == "interval":
                inc = [[data[t][0] + param[t], data[t][1] + param[t]] for t in np.arange(0, n)]
            elif type == "distribution":
                for t in np.arange(0, n):
                    data[t].differential_offset(param[t])
                inc = data
        else:
            if type == "point":
                inc = [data[0] + param[0]]
                for t in np.arange(1, steps_ahead):
                    inc.append(data[t] + inc[t-1])
            elif type == "interval":
                inc = [[data[0][0] + param[0], data[0][1] + param[0]]]
                for t in np.arange(1, steps_ahead):
                    inc.append([data[t][0] + np.nanmean(inc[t-1]), data[t][1] + np.nanmean(inc[t-1])])
            elif type == "distribution":
                data[0].differential_offset(param[0])
                for t in np.arange(1, steps_ahead):
                    ex = data[t-1].expected_value()
                    data[t].differential_offset(ex)
                inc = data

        if n == 1:
            return inc[0]
        else:
            return inc


class Scale(Transformation):
    """
    Scale data inside a interval [min, max]

    
    """
    def __init__(self, min=0, max=1):
        super(Scale, self).__init__()
        self.data_max = None
        self.data_min = None
        self.transf_max = max
        self.transf_min = min
        self.name = 'Scale'

    @property
    def parameters(self):
        return [self.transf_max, self.transf_min]

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
        self.name = 'AdaptExpect'

    @property
    def parameters(self):
        return self.parameters

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
    """
    Box-Cox power transformation

    y'(t) = log( y(t) )
    y(t) = exp( y'(t) )
    """
    def __init__(self, plambda):
        super(BoxCox, self).__init__()
        self.plambda = plambda
        self.name = 'BoxCox'

    @property
    def parameters(self):
        return self.plambda

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


class ROI(Transformation):
    """
    Return of Investment (ROI) transformation. Retrieved from Sadaei and Lee (2014) - Multilayer Stock
    Forecasting Model Using Fuzzy Time Series

    y'(t) = ( y(t) - y(t-1) ) / y(t-1)
    y(t) = ( y(t-1) * y'(t) ) + y(t-1)
    """
    def __init__(self, **kwargs):
        super(ROI, self).__init__()
        self.name = 'ROI'

    def apply(self, data, param=None, **kwargs):
        modified = [(data[i] - data[i - 1]) / data[i - 1] for i in np.arange(1, len(data))]
        modified.insert(0, .0)
        return modified

    def inverse(self, data, param=None, **kwargs):
        modified = [(param[i - 1] * data[i]) + param[i - 1] for i in np.arange(1, len(data))]
        return modified


class LinearTrend(Transformation):
    """
    Linear Trend. Estimate

    y'(t) = y(t) - (a*t+b)
    y(t) =  y'(t) + (a*t+b)
    """
    def __init__(self, **kwargs):
        super(LinearTrend, self).__init__()
        self.name = 'LinearTrend'
        self.index_type = kwargs.get('index_type','linear')
        '''The type of the time index used to train the regression coefficients. Available types are: field, datetime'''
        self.index_field = kwargs.get('index_field', None)
        '''The Pandas Dataframe column to use as index'''
        self.data_field = kwargs.get('data_field', None)
        '''The Pandas Dataframe column to use as data'''
        self.datetime_mask = kwargs.get('datetime_mask', None)
        '''The Pandas Dataframe mask for datetime indexes '''

        self.model = None
        '''Regression model'''

    def train(self, data, **kwargs):
        from pandas import datetime
        from sklearn.linear_model import LinearRegression

        x = data[self.index_field].values

        if self.index_type == 'datetime':
            x = pd.to_numeric(x, downcast='integer')

        indexes = np.reshape(x, (len(x), 1))
        values = data[self.data_field].values
        self.model = LinearRegression()
        self.model.fit(indexes, values)

    def trend(self, data):
        x = data[self.index_field].values
        if self.index_type == 'datetime':
            x = pd.to_numeric(x, downcast='integer')
        indexes = np.reshape(x, (len(x), 1))
        _trend = self.model.predict(indexes)
        return _trend

    def apply(self, data, param=None, **kwargs):
        values = data[self.data_field].values
        _trend = self.trend(data)
        modified = values - _trend
        return modified

    def inverse(self, data, param=None, **kwargs):
        x = self.generate_indexes(data, param[self.index_field].values[0], **kwargs)
        indexes = np.reshape(x, (len(x), 1))
        _trend = self.model.predict(indexes)
        modified = data + _trend
        return modified

    def increment(self,value, **kwargs):
        if self.index_type == 'linear':
            return value + 1
        elif self.index_type == 'datetime':
            if 'date_offset' not in kwargs:
                raise Exception('A pandas.DateOffset must be passed in the parameter ''date_offset''')
            doff = kwargs.get('date_offset')
            return value + doff

    def generate_indexes(self, data, value, **kwargs):
        if self.index_type == 'datetime':
            ret = [self.increment(pd.to_datetime(value, format=self.datetime_mask), **kwargs)]
        else:
            ret = [self.increment(value, **kwargs)]
        for i in np.arange(1,len(data)):
            ret.append(self.increment(ret[-1], **kwargs))

        if self.index_type == 'datetime':
            ret = pd.Series(ret)
            ret = pd.to_numeric(ret, downcast='integer')

        return np.array(ret)
