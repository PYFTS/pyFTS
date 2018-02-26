#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA as stats_arima
import scipy.stats as st
from pyFTS.common import SortedCollection, fts
from pyFTS.probabilistic import ProbabilityDistribution


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
        self.has_probability_forecasting = True
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

        if self.indexer is not None:
            data = self.indexer.get_data(data)

        data = self.apply_transformations(data, updateUoD=True)

        old_fit = self.model_fit
        try:
            self.model =  stats_arima(data, order=(self.p, self.d, self.q))
            self.model_fit = self.model.fit(disp=0)
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

        if self.indexer is not None and isinstance(data, pd.DataFrame):
            data = self.indexer.get_data(data)

        ndata = np.array(self.apply_transformations(data))

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

        ret = self.apply_inverse_transformations(ret, params=[data[self.order - 1:]])

        return ret

    def forecast_interval(self, data, **kwargs):

        if self.model_fit is None:
            return np.nan

        sigma = np.sqrt(self.model_fit.sigma2)

        #ndata = np.array(self.apply_transformations(data))

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

        #ret = self.apply_inverse_transformations(ret, params=[data[self.order - 1:]], point_to_interval=True)

        return ret

    def forecast_ahead_interval(self, data, steps, **kwargs):
        if self.model_fit is None:
            return np.nan

        smoothing = kwargs.get("smoothing",0.5)

        sigma = np.sqrt(self.model_fit.sigma2)

        ndata = np.array(self.apply_transformations(data))

        l = len(ndata)

        nmeans = self.forecast_ahead(ndata, steps, **kwargs)

        ret = []

        for k in np.arange(0, steps):
            tmp = []

            hsigma = (1 + k*smoothing)*sigma

            tmp.append(nmeans[k] + st.norm.ppf(self.alpha) * hsigma)
            tmp.append(nmeans[k] + st.norm.ppf(1 - self.alpha) * hsigma)

            ret.append(tmp)

        ret = self.apply_inverse_transformations(ret, params=[[data[-1] for a in np.arange(0, steps)]], interval=True)

        return ret

    def empty_grid(self, resolution):
        return self.get_empty_grid(-(self.original_max*2), self.original_max*2, resolution)

    def forecast_distribution(self, data, **kwargs):

        if self.indexer is not None and isinstance(data, pd.DataFrame):
            data = self.indexer.get_data(data)

        sigma = np.sqrt(self.model_fit.sigma2)

        l = len(data)

        ret = []

        for k in np.arange(self.order, l + 1):
            tmp = []

            sample = [data[i] for i in np.arange(k - self.order, k)]

            mean = self.forecast(sample)

            if isinstance(mean, (list, np.ndarray)):
                mean = mean[0]

            dist = ProbabilityDistribution.ProbabilityDistribution(type="histogram", uod=[self.original_min, self.original_max])
            intervals = []
            for alpha in np.arange(0.05, 0.5, 0.05):

                qt1 = mean + st.norm.ppf(alpha) * sigma
                qt2 = mean + st.norm.ppf(1 - alpha) * sigma

                intervals.append([qt1, qt2])

            dist.append_interval(intervals)

            ret.append(dist)

        return ret


    def forecast_ahead_distribution(self, data, steps, **kwargs):
        smoothing = kwargs.get("smoothing", 0.5)

        sigma = np.sqrt(self.model_fit.sigma2)

        l = len(data)

        ret = []

        nmeans = self.forecast_ahead(data, steps, **kwargs)

        for k in np.arange(0, steps):
            dist = ProbabilityDistribution.ProbabilityDistribution(type="histogram",
                                                                   uod=[self.original_min, self.original_max])
            intervals = []
            for alpha in np.arange(0.05, 0.5, 0.05):
                tmp = []

                hsigma = (1 + k * smoothing) * sigma

                tmp.append(nmeans[k] + st.norm.ppf(alpha) * hsigma)
                tmp.append(nmeans[k] + st.norm.ppf(1 - alpha) * hsigma)

                intervals.append(tmp)

            dist.append_interval(intervals)

            ret.append(dist)

        return ret