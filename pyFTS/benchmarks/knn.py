#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
from statsmodels.tsa.tsatools import lagmat
from pyFTS.common import fts
from pyFTS.probabilistic import ProbabilityDistribution


class KNearestNeighbors(fts.FTS):
    """
    K-Nearest Neighbors
    """
    def __init__(self, name, **kwargs):
        super(KNearestNeighbors, self).__init__(1, "kNN"+name)
        self.name = "kNN"
        self.shortname = "kNN"
        self.detail = "K-Nearest Neighbors"
        self.is_high_order = True
        self.has_point_forecasting = True
        self.has_interval_forecasting = True
        self.has_probability_forecasting = True
        self.benchmark_only = True
        self.min_order = 1
        self.alpha = kwargs.get("alpha", 0.05)
        self.order = kwargs.get("order", 1)
        self.lag = None
        self.k = kwargs.get("k", 30)
        self.uod = None

    def train(self, data, **kwargs):
        if kwargs.get('order', None) is not None:
            self.order = kwargs.get('order', 1)

        self.data = np.array(data)
        self.original_max = max(data)
        self.original_min = min(data)

        #self.lagdata, = lagmat(data, maxlag=self.order, trim="both", original='sep')


    def knn(self, sample):

        if self.order == 1:
            dist = np.apply_along_axis(lambda x: (x - sample) ** 2, 0, self.data)
            ix = np.argsort(dist) + 1
        else:
            dist = []
            for k in np.arange(self.order, len(self.data)):
                dist.append(sum([ (self.data[k - kk] - sample[kk])**2 for kk in range(self.order)]))
            ix = np.argsort(np.array(dist)) + self.order + 1

        ix2 = np.clip(ix[:self.k], 0, len(self.data)-1)
        return self.data[ix2]

    def forecast_distribution(self, data, **kwargs):
        ret = []

        smooth = kwargs.get("smooth", "KDE")
        alpha = kwargs.get("alpha", None)

        uod = self.get_UoD()

        for k in np.arange(self.order, len(data)):

            sample = data[k-self.order : k]

            forecasts = self.knn(sample)

            dist = ProbabilityDistribution.ProbabilityDistribution(smooth, uod=uod, data=forecasts,
                                                                   name="", **kwargs)
            ret.append(dist)

        return ret


