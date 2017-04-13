#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
from statsmodels.tsa.arima_model import ARIMA as stats_arima
from statsmodels.tsa.arima_model import ARMA
from pyFTS import fts


class ARIMA(fts.FTS):
    def __init__(self, name):
        super(ARIMA, self).__init__(1, "ARIMA")
        self.name = "ARIMA"
        self.detail = "Auto Regressive Integrated Moving Average"
        self.isHighOrder = True
        self.model = None
        self.model_fit = None
        self.trained_data = None
        self.p = 1
        self.d = 0
        self.q = 0
        self.benchmark_only = True
        self.minOrder = 1

    def train(self, data, sets, order=1, parameters=None):
        ndata = np.array(self.doTransformations(data))

        if parameters is not None:
            self.p = parameters[0]
            self.d = parameters[1]
            self.q = parameters[2]
            self.order = max([self.p, self.d, self.q])
            self.shortname = "ARIMA(" + str(self.p) + "," + str(self.d) + "," + str(self.q) + ")"

        old_fit = self.model_fit
        self.model =  stats_arima(ndata, order=(self.p, self.d, self.q))
        #try:
        self.model_fit = self.model.fit(disp=0)
        #except:
        #    try:
        #        self.model = stats_arima(data, order=(self.p, self.d, self.q))
        #        self.model_fit = self.model.fit(disp=1)
        #    except:
        #        self.model_fit = old_fit

        #self.trained_data = data #.tolist()

    def ar(self, data):
        return data.dot(self.model_fit.arparams)

    def ma(self, data):
        return data.dot(self.model_fit.maparams)

    def forecast(self, data):
        if self.model_fit is None:
            return np.nan

        order = self.p

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)

        ret = []

        ar = np.array([self.ar(ndata[k - self.p: k]) for k in np.arange(self.p, l)])

        residuals = np.array([ar[k - self.p] - ndata[k] for k in np.arange(self.p, l)])

        ma = np.array([self.ma(residuals[k - self.q : k]) for k in np.arange(self.q, len(ar)+1)])

        ret = ar + ma

        ret = self.doInverseTransformations(ret, params=[data[order - 1:]])

        return ret
