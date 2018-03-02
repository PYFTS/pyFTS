#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
from pyFTS.common import SortedCollection, fts, tree
from pyFTS.models import chen, cheng, hofts, hwang, ismailefendi, sadaei, song, yu
import scipy.stats as st


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
        self.order = 1
        self.point_method = kwargs.get('point_method', 'mean')
        self.interval_method = kwargs.get('interval_method', 'quantile')

    def append_model(self, model):
        self.models.append(model)
        if model.order > self.order:
            self.order = model.order

    def train(self, data, **kwargs):
        self.original_max = max(data)
        self.original_min = min(data)

    def get_models_forecasts(self,data):
        tmp = []
        for model in self.models:
            if self.is_multivariate or self.has_seasonality:
                forecast = model.forecast(data)
            else:
                sample = data[-model.order:]
                forecast = model.forecast(sample)
                if isinstance(forecast, (list,np.ndarray)) and len(forecast) > 0:
                    forecast = int(forecast[-1])
                elif isinstance(forecast, (list,np.ndarray)) and len(forecast) == 0:
                    forecast = np.nan
            if isinstance(forecast, list):
                tmp.extend(forecast)
            else:
                tmp.append(forecast)
        return tmp

    def get_point(self,forecasts, **kwargs):
        if self.point_method == 'mean':
            ret = np.nanmean(forecasts)
        elif self.point_method == 'median':
            ret = np.nanpercentile(forecasts, 50)
        elif self.point_method == 'quantile':
            alpha = kwargs.get("alpha",0.05)
            ret = np.percentile(forecasts, alpha*100)

        return ret

    def get_interval(self, forecasts):
        ret = []
        if self.interval_method == 'extremum':
            ret.append([min(forecasts), max(forecasts)])
        elif self.interval_method == 'quantile':
            qt_lo = np.nanpercentile(forecasts, q=self.alpha * 100)
            qt_up = np.nanpercentile(forecasts, q=(1-self.alpha) * 100)
            ret.append([qt_lo, qt_up])
        elif self.interval_method == 'normal':
            mu = np.nanmean(forecasts)
            sigma = np.sqrt(np.nanvar(forecasts))
            ret.append(mu + st.norm.ppf(self.alpha) * sigma)
            ret.append(mu + st.norm.ppf(1 - self.alpha) * sigma)

        return ret

    def get_distribution_interquantile(self,forecasts, alpha):
        size = len(forecasts)
        qt_lower = int(np.ceil(size * alpha)) - 1
        qt_upper = int(np.ceil(size * (1- alpha))) - 1

        ret = sorted(forecasts)[qt_lower : qt_upper]

        return ret

    def forecast(self, data, **kwargs):

        if "method" in kwargs:
            self.point_method = kwargs.get('method','mean')

        l = len(data)
        ret = []

        for k in np.arange(self.order, l+1):
            sample = data[k - self.order : k]
            tmp = self.get_models_forecasts(sample)
            point = self.get_point(tmp)
            ret.append(point)

        return ret

    def forecast_interval(self, data, **kwargs):

        if "method" in kwargs:
            self.interval_method = kwargs.get('method','quantile')

        if 'alpha' in kwargs:
            self.alpha = kwargs.get('alpha',0.05)

        l = len(data)

        ret = []

        for k in np.arange(self.order, l+1):
            sample = data[k - self.order : k]
            tmp = self.get_models_forecasts(sample)
            interval = self.get_interval(tmp)
            if len(interval) == 1:
                interval = interval[-1]
            ret.append(interval)

        return ret

    def forecast_ahead_interval(self, data, steps, **kwargs):

        if 'method' in kwargs:
            self.interval_method = kwargs.get('method','quantile')

        if 'alpha' in kwargs:
            self.alpha = kwargs.get('alpha', self.alpha)

        ret = []

        samples = [[k] for k in data[-self.order:]]

        for k in np.arange(self.order, steps + self.order):
            forecasts = []
            lags = {}
            for i in np.arange(0, self.order): lags[i] = samples[k - self.order + i]

            # Build the tree with all possible paths

            root = tree.FLRGTreeNode(None)

            tree.build_tree_without_order(root, lags, 0)

            for p in root.paths():
                path = list(reversed(list(filter(None.__ne__, p))))

                forecasts.extend(self.get_models_forecasts(path))

            samples.append(sampler(forecasts, np.arange(0.1, 1, 0.2)))
            interval = self.get_interval(forecasts)

            if len(interval) == 1:
                interval = interval[0]

            ret.append(interval)

        return ret

    def empty_grid(self, resolution):
        return self.get_empty_grid(-(self.original_max*2), self.original_max*2, resolution)

    def forecast_ahead_distribution(self, data, steps, **kwargs):
        if 'method' in kwargs:
            self.point_method = kwargs.get('method','mean')

        percentile_size = (self.original_max - self.original_min) / 100

        resolution = kwargs.get('resolution', percentile_size)

        grid = self.empty_grid(resolution)

        index = SortedCollection.SortedCollection(iterable=grid.keys())

        ret = []

        samples = [[k] for k in data[-self.order:]]

        for k in np.arange(self.order, steps + self.order):
            forecasts = []
            lags = {}
            for i in np.arange(0, self.order): lags[i] = samples[k - self.order + i]

            # Build the tree with all possible paths

            root = tree.FLRGTreeNode(None)

            tree.build_tree_without_order(root, lags, 0)

            for p in root.paths():
                path = list(reversed(list(filter(None.__ne__, p))))

                forecasts.extend(self.get_models_forecasts(path))

            samples.append(sampler(forecasts, np.arange(0.1, 1, 0.1)))

            grid = self.gridCountPoint(grid, resolution, index, forecasts)

            tmp = np.array([grid[i] for i in sorted(grid)])

            ret.append(tmp / sum(tmp))

        grid = self.empty_grid(resolution)
        df = pd.DataFrame(ret, columns=sorted(grid))
        return df


class AllMethodEnsembleFTS(EnsembleFTS):
    def __init__(self, **kwargs):
        super(AllMethodEnsembleFTS, self).__init__(name="Ensemble FTS", **kwargs)
        self.min_order = 3

    def set_transformations(self, model):
        for t in self.transformations:
            model.append_transformation(t)

    def train(self, data, **kwargs):
        self.original_max = max(data)
        self.original_min = min(data)

        order = kwargs.get('order',2)

        fo_methods = [song.ConventionalFTS, chen.ConventionalFTS, yu.WeightedFTS, cheng.TrendWeightedFTS,
                      sadaei.ExponentialyWeightedFTS, ismailefendi.ImprovedWeightedFTS]

        ho_methods = [hofts.HighOrderFTS, hwang.HighOrderFTS]

        for method in fo_methods:
            model = method("")
            self.set_transformations(model)
            model.train(data, **kwargs)
            self.append_model(model)

        for method in ho_methods:
            for o in np.arange(1, order+1):
                model = method("")
                if model.min_order >= o:
                    self.set_transformations(model)
                    model.train(data, **kwargs)
                    self.append_model(model)




