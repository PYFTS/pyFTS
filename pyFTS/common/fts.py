import numpy as np
import pandas as pd
from pyFTS.common import FuzzySet, SortedCollection, tree


class FTS(object):
    """
    Fuzzy Time Series
    """
    def __init__(self, order, name, **kwargs):
        """
        Create a Fuzzy Time Series model
        :param order: model order
        :param name: model name
        :param kwargs: model specific parameters
        """
        self.sets = {}
        self.flrgs = {}
        self.order = order
        self.shortname = name
        self.name = name
        self.detail = name
        self.is_high_order = False
        self.min_order = 1
        self.has_seasonality = False
        self.has_point_forecasting = True
        self.has_interval_forecasting = False
        self.has_probability_forecasting = False
        self.is_multivariate = False
        self.dump = False
        self.transformations = []
        self.transformations_param = []
        self.original_max = 0
        self.original_min = 0
        self.partitioner = kwargs.get("partitioner", None)
        if self.partitioner != None:
            self.sets = self.partitioner.sets
        self.auto_update = False
        self.benchmark_only = False
        self.indexer = None

    def fuzzy(self, data):
        """
        Fuzzify a data point
        :param data: data point
        :return: maximum membership fuzzy set
        """
        best = {"fuzzyset": "", "membership": 0.0}

        for f in self.sets:
            fset = self.sets[f]
            if best["membership"] <= fset.membership(data):
                best["fuzzyset"] = fset.name
                best["membership"] = fset.membership(data)

        return best

    def predict(self, data, **kwargs):
        """
        Forecast using trained model
        :param data: time series with minimal length to the order of the model
        :param kwargs:
        :return:
        """
        type = kwargs.get("type", 'point')
        steps_ahead = kwargs.get("steps_ahead", None)

        if type == 'point' and steps_ahead == None:
            return self.forecast(data, **kwargs)
        elif type == 'point' and steps_ahead != None:
            return self.forecast_ahead(data, steps_ahead, **kwargs)
        elif type == 'interval' and steps_ahead == None:
            return self.forecast_interval(data, **kwargs)
        elif type == 'interval' and steps_ahead != None:
            return self.forecast_ahead_interval(data, steps_ahead, **kwargs)
        elif type == 'distribution' and steps_ahead == None:
            return self.forecast_distribution(data, **kwargs)
        elif type == 'distribution' and steps_ahead != None:
            return self.forecast_ahead_distribution(data, steps_ahead, **kwargs)
        else:
            raise ValueError('The argument \'type\' has an unknown value.')


    def forecast(self, data, **kwargs):
        """
        Point forecast one step ahead 
        :param data: time series with minimal length to the order of the model
        :param kwargs: 
        :return: 
        """
        raise NotImplementedError('This model do not perform one step ahead point forecasts!')

    def forecast_interval(self, data, **kwargs):
        """
        Interval forecast one step ahead
        :param data: 
        :param kwargs: 
        :return: 
        """
        raise NotImplementedError('This model do not perform one step ahead interval forecasts!')

    def forecast_distribution(self, data, **kwargs):
        """
        Probabilistic forecast one step ahead
        :param data: 
        :param kwargs: 
        :return: 
        """
        raise NotImplementedError('This model do not perform one step ahead distribution forecasts!')

    def forecast_ahead(self, data, steps, **kwargs):
        """
        Point forecast n steps ahead
        :param data: 
        :param steps: 
        :param kwargs: 
        :return: 
        """
        ret = []
        for k in np.arange(0,steps):
            tmp = self.forecast(data[-self.order:], **kwargs)

            if isinstance(tmp,(list, np.ndarray)):
                tmp = tmp[0]

            ret.append(tmp)
            data.append(tmp)

        return ret

    def forecast_ahead_interval(self, data, steps, **kwargs):
        """
        Interval forecast n steps ahead
        :param data: 
        :param steps: 
        :param kwargs: 
        :return: 
        """
        raise NotImplementedError('This model do not perform multi step ahead interval forecasts!')

    def forecast_ahead_distribution(self, data, steps, **kwargs):
        """
        Probabilistic forecast n steps ahead
        :param data: 
        :param steps: 
        :param kwargs: 
        :return: 
        """
        raise NotImplementedError('This model do not perform multi step ahead distribution forecasts!')

    def train(self, data, **kwargs):
        """
        
        :param data: 
        :param sets: 
        :param order: 
        :param parameters: 
        :return: 
        """
        pass

    def fit(self, data, **kwargs):
        """

        :param data:
        :param kwargs:
        :return:
        """
        self.train(data, **kwargs)

    def append_transformation(self, transformation):
        if transformation is not None:
            self.transformations.append(transformation)

    def apply_transformations(self, data, params=None, updateUoD=False, **kwargs):
        ndata = data
        if updateUoD:
            if min(data) < 0:
                self.original_min = min(data) * 1.1
            else:
                self.original_min = min(data) * 0.9

            if max(data) > 0:
                self.original_max = max(data) * 1.1
            else:
                self.original_max = max(data) * 0.9

        if len(self.transformations) > 0:
            if params is None:
                params = [ None for k in self.transformations]

            for c, t in enumerate(self.transformations, start=0):
                ndata = t.apply(ndata,params[c])

        return ndata

    def apply_inverse_transformations(self, data, params=None, **kwargs):
        if len(self.transformations) > 0:
            if params is None:
                params = [None for k in self.transformations]

            for c, t in enumerate(reversed(self.transformations), start=0):
                ndata = t.inverse(data, params[c], **kwargs)

            return ndata
        else:
            return data

    def __str__(self):
        tmp = self.name + ":\n"
        for r in sorted(self.flrgs):
            tmp = tmp + str(self.flrgs[r]) + "\n"
        return tmp

    def __len__(self):
       return len(self.flrgs)

    def len_total(self):
        return sum([len(k) for k in self.flrgs])

    def get_empty_grid(self, _min, _max, resolution):
        grid = {}

        for sbin in np.arange(_min,_max, resolution):
            grid[sbin] = 0

        return grid

    def getGridClean(self, resolution):
        if len(self.transformations) == 0:
            _min = self.sets[0].lower
            _max = self.sets[-1].upper
        else:
            _min = self.original_min
            _max = self.original_max
        return self.get_empty_grid(_min, _max, resolution)



    def gridCount(self, grid, resolution, index, interval):
        #print(point_to_interval)
        for k in index.inside(interval[0],interval[1]):
            grid[k] += 1
        return grid

    def gridCountPoint(self, grid, resolution, index, point):
        if not isinstance(point, (list, np.ndarray)):
            point = [point]

        for p in point:
            k = index.find_ge(p)
            grid[k] += 1
        return grid

    def get_UoD(self):
        return [self.original_min, self.original_max]






