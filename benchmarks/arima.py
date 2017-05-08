#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
from statsmodels.tsa.arima_model import ARIMA as stats_arima
from pyFTS import fts


class ARIMA(fts.FTS):
    """
    Façade for statsmodels.tsa.arima_model
    """
    def __init__(self, name, **kwargs):
        super(ARIMA, self).__init__(1, "ARIMA")
        self.name = "ARIMA"
        self.detail = "Auto Regressive Integrated Moving Average"
        self.is_high_order = True
        self.model = None
        self.model_fit = None
        self.trained_data = None
        self.p = 1
        self.d = 0
        self.q = 0
        self.benchmark_only = True
        self.min_order = 1

    def train(self, data, sets, order=(2,1,1), parameters=None):
        self.p = order[0]
        self.d = order[1]
        self.q = order[2]
        self.order = max([self.p, self.d, self.q])
        self.shortname = "ARIMA(" + str(self.p) + "," + str(self.d) + "," + str(self.q) + ")"

        old_fit = self.model_fit
        self.model =  stats_arima(data, order=(self.p, self.d, self.q))
        self.model_fit = self.model.fit(disp=0)

    def ar(self, data):
        return data.dot(self.model_fit.arparams)

    def ma(self, data):
        return data.dot(self.model_fit.maparams)

    def forecast(self, data, **kwargs):
        if self.model_fit is None:
            return np.nan

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)

        ret = []

        ar = np.array([self.ar(ndata[k - self.p: k]) for k in np.arange(self.p, l)])

        residuals = np.array([ar[k - self.p] - ndata[k] for k in np.arange(self.p, l)])

        ma = np.array([self.ma(residuals[k - self.q: k]) for k in np.arange(self.q, len(ar) + 1)])

        ret = ar + ma

        ret = self.doInverseTransformations(ret, params=[data[self.order - 1:]])

        return ret