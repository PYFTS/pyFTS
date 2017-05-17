#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
import math
from operator import itemgetter
from pyFTS.common import FLR, FuzzySet, SortedCollection
from pyFTS import fts, chen, cheng, hofts, hwang, ismailefendi, sadaei, song, yu
from pyFTS.benchmarks import arima, quantreg
from pyFTS.common import Transformations
import scipy.stats as st
from pyFTS import tree


def sampler(data, quantiles):
    ret = []
    for qt in quantiles:
        ret.append(np.nanpercentile(data, q=qt * 100))
    return ret


class EnsembleFTS(fts.FTS):
    def __init__(self, name, **kwargs):
        super(EnsembleFTS, self).__init__(1, "Ensemble FTS")
        self.shortname = "Ensemble FTS " + name
        self.name = "Ensemble FTS"
        self.flrgs = {}
        self.has_point_forecasting = True
        self.has_interval_forecasting = True
        self.has_probability_forecasting = True
        self.is_high_order = True
        self.models = []
        self.parameters = []
        self.alpha = kwargs.get("alpha", 0.05)
        self.max_order = 1

    def appendModel(self, model):
        self.models.append(model)
        if model.order > self.max_order:
            self.max_order = model.order

    def train(self, data, sets, order=1,parameters=None):
        self.original_max = max(data)
        self.original_min = min(data)

    def get_models_forecasts(self,data):
        tmp = []
        for model in self.models:
            sample = data[-model.order:]
            forecast = model.forecast(sample)
            if isinstance(forecast, (list,np.ndarray)):
                forecast = int(forecast[-1])
            tmp.append(forecast)
        return tmp

    def get_point(self,method, forecasts, **kwargs):
        if method == 'mean':
            ret = np.nanmean(forecasts)
        elif method == 'median':
            ret = np.nanpercentile(forecasts, 50)
        elif method == 'quantile':
            alpha = kwargs.get("alpha",0.05)
            ret = np.percentile(forecasts, alpha*100)

        return ret

    def get_interval(self, method, forecasts):
        ret = []
        if method == 'extremum':
            ret.append([min(forecasts), max(forecasts)])
        elif method == 'quantile':
            qt_lo = np.nanpercentile(forecasts, q=self.alpha * 100)
            qt_up = np.nanpercentile(forecasts, q=(1-self.alpha) * 100)
            ret.append([qt_lo, qt_up])
        elif method == 'normal':
            mu = np.nanmean(forecasts)
            sigma = np.sqrt(np.nanvar(forecasts))
            ret.append(mu + st.norm.ppf(self.alpha) * sigma)
            ret.append(mu + st.norm.ppf(1 - self.alpha) * sigma)

        return ret

    def forecast(self, data, **kwargs):

        method = kwargs.get('method','mean')

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)
        ret = []

        for k in np.arange(self.max_order, l+1):
            sample = ndata[k - self.max_order : k ]
            tmp = self.get_models_forecasts(sample)
            point = self.get_point(method, tmp)
            ret.append(point)

        ret = self.doInverseTransformations(ret, params=[data[self.order - 1:]])

        return ret


    def forecastInterval(self, data, **kwargs):

        method = kwargs.get('method', 'extremum')

        if 'alpha' in kwargs:
            self.alpha = kwargs.get('alpha',0.05)

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)

        ret = []

        for k in np.arange(self.max_order, l+1):
            sample = ndata[k - self.max_order : k ]
            tmp = self.get_models_forecasts(sample)
            interval = self.get_interval(method, tmp)
            ret.append(interval)

        return ret

    def forecastAheadInterval(self, data, steps, **kwargs):

        method = kwargs.get('method', 'extremum')

        ret = []

        samples = [[k,k] for k in data[-self.max_order:]]

        for k in np.arange(self.max_order, steps):
            forecasts = []
            sample = samples[k - self.max_order : k]
            lo_sample = [i[0] for i in sample]
            up_sample = [i[1] for i in sample]
            forecasts.extend(self.get_models_forecasts(lo_sample) )
            forecasts.extend(self.get_models_forecasts(up_sample))
            interval = self.get_interval(method, forecasts)

            if len(interval) == 1:
                interval = interval[0]

            ret.append(interval)
            samples.append(interval)

        return ret

    def empty_grid(self, resolution):
        return self.get_empty_grid(-(self.original_max*2), self.original_max*2, resolution)

    def forecastAheadDistribution(self, data, steps, **kwargs):
        method = kwargs.get('method', 'extremum')

        percentile_size = (self.original_max - self.original_min) / 100

        resolution = kwargs.get('resolution', percentile_size)

        grid = self.empty_grid(resolution)

        index = SortedCollection.SortedCollection(iterable=grid.keys())

        ret = []

        samples = [[k] for k in data[-self.max_order:]]

        for k in np.arange(self.max_order, steps + self.max_order):
            forecasts = []
            lags = {}
            for i in np.arange(0, self.max_order): lags[i] = samples[k - self.max_order + i]

            # Build the tree with all possible paths

            root = tree.FLRGTreeNode(None)

            tree.buildTreeWithoutOrder(root, lags, 0)

            for p in root.paths():
                path = list(reversed(list(filter(None.__ne__, p))))

                forecasts.extend(self.get_models_forecasts(path))

            samples.append(sampler(forecasts, [0.05, 0.25, 0.5, 0.75, 0.95 ]))

            grid = self.gridCountPoint(grid, resolution, index, forecasts)

            tmp = np.array([grid[i] for i in sorted(grid)])

            ret.append(tmp / sum(tmp))

        return ret

