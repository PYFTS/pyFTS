#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
from statsmodels.tsa.arima_model import ARIMA as stats_arima
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
        if parameters is not None:
            self.p = parameters[0]
            self.d = parameters[1]
            self.q = parameters[2]
            self.order = max([self.p, self.d, self.q])
            self.shortname = "ARIMA(" + str(self.p) + "," + str(self.d) + "," + str(self.q) + ")"

        old_fit = self.model_fit
        self.model =  stats_arima(data, order=(self.p, self.d, self.q))
        try:
            self.model_fit = self.model.fit(disp=0)
        except:
            try:
                self.model = stats_arima(data, order=(self.p, self.d, self.q))
                self.model_fit = self.model.fit(disp=1)
            except:
                self.model_fit = old_fit

        self.trained_data = data #.tolist()

    def forecast(self, data):
        ret = []
        for t in data:
            output = self.model_fit.forecast()
            ret.append( output[0] )
            self.trained_data = np.append(self.trained_data, t) #.append(t)
            self.train(self.trained_data,None,order=self.order, parameters=(self.p, self.d, self.q))
        return ret