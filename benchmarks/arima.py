#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA as stats_arima
import scipy.stats as st
from pyFTS import fts
from pyFTS.common import SortedCollection


class ARIMA(fts.FTS):
    """
    Façade for statsmodels.tsa.arima_model
    """
    def __init__(self, name, **kwargs):
        super(ARIMA, self).__init__(1, "ARIMA"+name)
        self.name = "ARIMA"
        self.detail = "Auto Regressive Integrated Moving Average"
        self.is_high_order = True
        self.has_point_forecasting = True
        self.has_interval_forecasting = True
        self.model = None
        self.model_fit = None
        self.trained_data = None
        self.p = 1
        self.d = 0
        self.q = 0
        self.benchmark_only = True
        self.min_order = 1
        self.alpha = kwargs.get("alpha", 0.05)
        self.shortname += str(self.alpha)

    def train(self, data, sets, order, parameters=None):
        self.p = order[0]
        self.d = order[1]
        self.q = order[2]
        self.order = self.p + self.q
        self.shortname = "ARIMA(" + str(self.p) + "," + str(self.d) + "," + str(self.q) + ") - " + str(self.alpha)

        data = self.doTransformations(data, updateUoD=True)

        old_fit = self.model_fit
        try:
            self.model =  stats_arima(data, order=(self.p, self.d, self.q))
            self.model_fit = self.model.fit(disp=0)
            print(np.sqrt(self.model_fit.sigma2))
        except Exception as ex:
            print(ex)
            self.model_fit = None

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

        if self.d == 0:
            ar = np.array([self.ar(ndata[k - self.p: k]) for k in np.arange(self.p, l+1)]) #+1 to forecast one step ahead given all available lags
        else:
            ar = np.array([ndata[k] + self.ar(ndata[k - self.p: k]) for k in np.arange(self.p, l+1)])

        if self.q > 0:
            residuals = np.array([ndata[k] - ar[k - self.p] for k in np.arange(self.p, l)])

            ma = np.array([self.ma(residuals[k - self.q: k]) for k in np.arange(self.q, len(residuals)+1)])

            ret = ar[self.q:] + ma
        else:
            ret = ar

        ret = self.doInverseTransformations(ret, params=[data[self.order - 1:]])

        return ret

    def forecastInterval(self, data, **kwargs):

        if self.model_fit is None:
            return np.nan

        sigma = np.sqrt(self.model_fit.sigma2)

        #ndata = np.array(self.doTransformations(data))

        l = len(data)

        ret = []

        for k in np.arange(self.order, l+1):
            tmp = []

            sample = [data[i] for i in np.arange(k - self.order, k)]

            mean = self.forecast(sample)

            if isinstance(mean,(list, np.ndarray)):
                mean = mean[0]

            tmp.append(mean + st.norm.ppf(self.alpha) * sigma)
            tmp.append(mean + st.norm.ppf(1 - self.alpha) * sigma)

            ret.append(tmp)

        #ret = self.doInverseTransformations(ret, params=[data[self.order - 1:]], point_to_interval=True)

        return ret

    def forecastAheadInterval(self, data, steps, **kwargs):
        if self.model_fit is None:
            return np.nan

        smoothing = kwargs.get("smoothing",0.5)

        sigma = np.sqrt(self.model_fit.sigma2)

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)

        nmeans = self.forecastAhead(ndata, steps, **kwargs)

        ret = []

        for k in np.arange(0, steps):
            tmp = []

            hsigma = (1 + k*smoothing)*sigma

            tmp.append(nmeans[k] + st.norm.ppf(self.alpha) * hsigma)
            tmp.append(nmeans[k] + st.norm.ppf(1 - self.alpha) * hsigma)

            ret.append(tmp)

        ret = self.doInverseTransformations(ret, params=[[data[-1] for a in np.arange(0,steps)]], interval=True)

        return ret

    def forecastAheadDistribution(self, data, steps, **kwargs):
        smoothing = kwargs.get("smoothing", 0.5)

        sigma = np.sqrt(self.model_fit.sigma2)

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)

        percentile_size = (self.original_max - self.original_min)/100

        resolution = kwargs.get('resolution', percentile_size)

        grid = self.get_empty_grid(self.original_min, self.original_max, resolution)

        index = SortedCollection.SortedCollection(iterable=grid.keys())

        ret = []

        nmeans = self.forecastAhead(ndata, steps, **kwargs)

        for k in np.arange(0, steps):
            grid = self.get_empty_grid(self.original_min, self.original_max, resolution)
            for alpha in np.arange(0.05, 0.5, 0.05):
                tmp = []

                hsigma = (1 + k * smoothing) * sigma

                tmp.append(nmeans[k] + st.norm.ppf(alpha) * hsigma)
                tmp.append(nmeans[k] + st.norm.ppf(1 - alpha) * hsigma)

                grid = self.gridCount(grid, resolution, index, tmp)

            tmp = np.array([grid[i] for i in sorted(grid)])

            ret.append(tmp / sum(tmp))

        grid = self.get_empty_grid(self.original_min, self.original_max, resolution)
        df = pd.DataFrame(ret, columns=sorted(grid))
        return df