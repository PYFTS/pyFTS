from pyFTS.common.transformations.transformation import Transformation 
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


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
