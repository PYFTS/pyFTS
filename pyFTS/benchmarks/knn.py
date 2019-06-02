#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
from statsmodels.tsa.tsatools import lagmat
from pyFTS.common import fts
from pyFTS.probabilistic import ProbabilityDistribution
from sklearn.neighbors import KDTree
from itertools import product
from pyFTS.models.ensemble.ensemble import sampler

class KNearestNeighbors(fts.FTS):
    """
    A façade for sklearn.neighbors
    """
    def __init__(self, **kwargs):
        super(KNearestNeighbors, self).__init__(**kwargs)
        self.name = "kNN"
        self.shortname = "kNN"
        self.detail = "K-Nearest Neighbors"
        self.uod_clip = False
        self.is_high_order = True
        self.has_point_forecasting = True
        self.has_interval_forecasting = True
        self.has_probability_forecasting = True
        self.benchmark_only = True
        self.min_order = 1
        self.alpha = kwargs.get("alpha", 0.05)
        self.lag = None
        self.k = kwargs.get("k", 30)
        self.uod = None
        self.kdtree = None
        self.values = None

    def _prepare_x(self, data):
        l = len(data)
        X = []

        if l == self.order:
            l += 1

        for t in np.arange(self.order, l):
            X.append([data[t - k - 1] for k in np.arange(self.order)])

        return X

    def _prepare_xy(self, data):
        l = len(data)
        X = []
        Y = []

        for t in np.arange(self.order, l):
            X.append([data[t - k - 1] for k in np.arange(self.order)])
            Y.append(data[t])

        return (X,Y)

    def train(self, data, **kwargs):
        X,Y = self._prepare_xy(data)

        self.kdtree = KDTree(np.array(X))
        self.values = Y

    def knn(self, sample):
        X = self._prepare_x(sample)
        _, ix = self.kdtree.query(np.array(X), self.k)

        return [self.values[k] for k in ix.flatten() ]

    def forecast(self, data, **kwargs):
        ret = []
        for k in np.arange(self.order, len(data)):

            sample = data[k-self.order : k]

            forecasts = self.knn(sample)

            ret.append(np.nanmean(forecasts))

        return ret

    def forecast_ahead(self, data, steps, **kwargs):
        start = kwargs.get('start', self.order)

        sample = [k for k in data[start - self.order: start]]

        for k in np.arange(self.order, steps + self.order):
            tmp = self.forecast(sample[k-self.order:k])
            sample.append(tmp)

        return sample[-steps]

    def forecast_interval(self, data, **kwargs):

        alpha = kwargs.get('alpha',self.alpha)

        ret = []
        for k in np.arange(self.order, len(data)):

            sample = data[k-self.order : k]

            forecasts = self.knn(sample)

            i = np.percentile(forecasts, [alpha*100, (1-alpha)*100]).tolist()
            ret.append(i)

        return ret

    def forecast_ahead_interval(self, data, steps, **kwargs):
        alpha = kwargs.get('alpha', self.alpha)

        ret = []

        start = kwargs.get('start', self.order)

        sample = [[k] for k in data[start - self.order: start]]

        for k in np.arange(self.order, steps + self.order):
            forecasts = []

            lags = [sample[k - i - 1] for i in np.arange(0, self.order)]

            # Trace the possible paths
            for path in product(*lags):
                forecasts.extend(self.knn(path))

            sample.append(sampler(forecasts, np.arange(.1, 1, 0.1), bounds=True))

            interval = np.percentile(forecasts, [alpha*100, (1-alpha)*100]).tolist()

            ret.append(interval)

        return ret

    def forecast_distribution(self, data, **kwargs):
        ret = []

        smooth = kwargs.get("smooth", "histogram")

        uod = self.get_UoD()

        for k in np.arange(self.order, len(data)):

            sample = data[k-self.order : k]

            forecasts = self.knn(sample)

            dist = ProbabilityDistribution.ProbabilityDistribution(smooth, uod=uod, data=forecasts,
                                                                   name="", **kwargs)
            ret.append(dist)

        return ret

    def forecast_ahead_distribution(self, data, steps, **kwargs):
        smooth = kwargs.get("smooth", "histogram")

        ret = []

        start = kwargs.get('start', self.order)

        uod = self.get_UoD()

        sample = [[k] for k in data[start - self.order: start]]

        for k in np.arange(self.order, steps + self.order):
            forecasts = []

            lags = [sample[k - i - 1] for i in np.arange(0, self.order)]

            # Trace the possible paths
            for path in product(*lags):
                forecasts.extend(self.knn(path))

            dist = ProbabilityDistribution.ProbabilityDistribution(smooth, uod=uod, data=forecasts,
                                                                   name="", **kwargs)
            ret.append(dist)

            sample.append(sampler(forecasts, np.arange(.1, 1, 0.1), bounds=True))

        return ret


