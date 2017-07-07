import numpy as np
import pandas as pd
from pyFTS import tree
from pyFTS.common import FuzzySet, SortedCollection


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
        self.partitioner = None
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

    def forecast(self, data, **kwargs):
        """
        Point forecast one step ahead 
        :param data: time series with minimal length to the order of the model
        :param kwargs: 
        :return: 
        """
        pass

    def forecastInterval(self, data, **kwargs):
        """
        Interval forecast one step ahead
        :param data: 
        :param kwargs: 
        :return: 
        """
        pass

    def forecastDistribution(self, data, **kwargs):
        """
        Probabilistic forecast one step ahead
        :param data: 
        :param kwargs: 
        :return: 
        """
        pass

    def forecastAhead(self, data, steps, **kwargs):
        """
        Point forecast n steps ahead
        :param data: 
        :param steps: 
        :param kwargs: 
        :return: 
        """
        ndata = [k for k in self.doTransformations(data[- self.order:])]

        ret = []
        for k in np.arange(0,steps):
            tmp = self.forecast(ndata[-self.order:], **kwargs)

            if isinstance(tmp,(list, np.ndarray)):
                tmp = tmp[0]

            ret.append(tmp)
            ndata.append(tmp)

        ret = self.doInverseTransformations(ret, params=[ndata[self.order - 1:]])

        return ret

    def forecastAheadInterval(self, data, steps, **kwargs):
        """
        Interval forecast n steps ahead
        :param data: 
        :param steps: 
        :param kwargs: 
        :return: 
        """
        pass

    def forecastAheadDistribution(self, data, steps, **kwargs):
        """
        Probabilistic forecast n steps ahead
        :param data: 
        :param steps: 
        :param kwargs: 
        :return: 
        """
        pass

    def train(self, data, sets, order=1, parameters=None):
        """
        
        :param data: 
        :param sets: 
        :param order: 
        :param parameters: 
        :return: 
        """
        pass

    def getMidpoints(self, flrg):
        ret = np.array([s.centroid for s in flrg.RHS])
        return ret

    def appendTransformation(self, transformation):
        if transformation is not None:
            self.transformations.append(transformation)

    def doTransformations(self,data,params=None,updateUoD=False, **kwargs):
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

    def doInverseTransformations(self, data, params=None, **kwargs):
        ndata = data
        if len(self.transformations) > 0:
            if params is None:
                params = [None for k in self.transformations]

            for c, t in enumerate(reversed(self.transformations), start=0):
                ndata = t.inverse(ndata, params[c], **kwargs)

        return ndata

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






