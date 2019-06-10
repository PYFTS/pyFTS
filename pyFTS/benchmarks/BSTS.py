#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
import pyflux as pf
import scipy.stats as st
from pyFTS.common import SortedCollection, fts
from pyFTS.probabilistic import ProbabilityDistribution


class ARIMA(fts.FTS):
    """
    Façade for statsmodels.tsa.arima_model
    """
    def __init__(self, **kwargs):
        super(ARIMA, self).__init__(**kwargs)
        self.name = "BSTS"
        self.detail = "Bayesian Structural Time Series"
        self.is_high_order = True
        self.has_point_forecasting = True
        self.has_interval_forecasting = True
        self.has_probability_forecasting = True
        self.uod_clip = False
        self.model = None
        self.model_fit = None
        self.trained_data = None
        self.p = 1
        self.d = 0
        self.q = 0
        self.benchmark_only = True
        self.min_order = 1
        self.alpha = kwargs.get("alpha", 0.05)
        self.order = kwargs.get("order", (1,0,0))
        self._decompose_order(self.order)
        self.model = None

    def _decompose_order(self, order):
        if isinstance(order, (tuple, set, list)):
            self.p = order[0]
            self.d = order[1]
            self.q = order[2]
            self.order = self.p + self.q + (self.q - 1 if self.q > 0 else 0)
            self.max_lag = self.order
            self.d = len(self.transformations)
            self.shortname = "BSTS({},{},{})-{}".format(self.p,self.d,self.q,self.alpha)

    def train(self, data, **kwargs):

        if 'order' in kwargs:
            order = kwargs.pop('order')
            self._decompose_order(order)

        if self.indexer is not None:
            data = self.indexer.get_data(data)

        try:
            self.model =  pf.ARIMA(data=data, ar=self.p, ma=self.q, integ=self.d, family=pf.Normal())
            self.model_fit = self.model.fit('M-H', nsims=20000)
        except Exception as ex:
            print(ex)
            self.model_fit = None

    def inference(self, steps):
        t_z = self.model.transform_z()
        mu, Y = self.model._model(self.model.latent_variables.get_z_values())
        date_index = self.model.shift_dates(steps)
        sim_vector = self.model._sim_prediction(mu, Y, steps, t_z, 1000)

        return sim_vector

    def forecast(self, ndata, **kwargs):
        raise NotImplementedError()

    def forecast_ahead(self, data, steps, **kwargs):
        return self.model.predict(steps, intervals=False).values.flatten().tolist()

    def forecast_interval(self, data, **kwargs):
        raise NotImplementedError()

    def forecast_ahead_interval(self, ndata, steps, **kwargs):
        sim_vector = self.inference(steps)

        if 'alpha' in kwargs:
            alpha = kwargs.get('alpha')
        else:
            alpha = self.alpha

        ret = []

        for ct, sample in enumerate(sim_vector):
            i = np.percentile(sample, [alpha*100, (1-alpha)*100]).tolist()
            ret.append(i)

        return ret

    def forecast_distribution(self, data, **kwargs):

        sim_vector = self.inference(steps)

        ret = []

        for ct, sample in enumerate(sim_vector):
            pd = ProbabilityDistribution.ProbabilityDistribution(type='histogram', data=sample, nbins=500)
            ret.append(pd)

        return ret


    def forecast_ahead_distribution(self, data, steps, **kwargs):

        sim_vector = self.inference(steps)

        ret = []

        for ct, sample in enumerate(sim_vector):
            pd = ProbabilityDistribution.ProbabilityDistribution(type='histogram', data=sample, nbins=500)
            ret.append(pd)

        return ret
