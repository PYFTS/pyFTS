#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
import pyflux as pf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import scipy.stats as st
from pyFTS.common import SortedCollection, fts
from pyFTS.probabilistic import ProbabilityDistribution


class GPR(fts.FTS):
    """
    Façade for sklearn.gaussian_proces
    """
    def __init__(self, **kwargs):
        super(GPR, self).__init__(**kwargs)
        self.name = "GPR"
        self.detail = "Gaussian Process Regression"
        self.is_high_order = True
        self.has_point_forecasting = True
        self.has_interval_forecasting = True
        self.has_probability_forecasting = True
        self.uod_clip = False
        self.benchmark_only = True
        self.min_order = 1
        self.alpha = kwargs.get("alpha", 0.05)

        self.lscale = kwargs.get('length_scale', 1)

        self.kernel = ConstantKernel(1.0) * RBF(length_scale=self.lscale)
        self.model = GaussianProcessRegressor(kernel=self.kernel, alpha=.05,
                                      n_restarts_optimizer=10,
                                      normalize_y=False)
        self.model_fit = None


    def train(self, data, **kwargs):
        l = len(data)
        X = []
        Y = []

        for t in np.arange(self.order, l):
            X.append([data[t - k - 1] for k in np.arange(self.order)])
            Y.append(data[t])

        self.model_fit = self.model.fit(X, Y)


    def forecast(self, data, **kwargs):
        X = []
        l = len(data)
        for t in np.arange(self.order, l):
            X.append([data[t - k - 1] for k in np.arange(self.order)])

        return self.model_fit.predict(X)

    def forecast_interval(self, data, **kwargs):

        if 'alpha' in kwargs:
            alpha = kwargs.get('alpha')
        else:
            alpha = self.alpha

        X = []
        l = len(data)
        for t in np.arange(self.order, l):
            X.append([data[t - k - 1] for k in np.arange(self.order)])

        Y, sigma = self.model_fit.predict(X, return_cov=True)

        uncertainty = st.norm.ppf(alpha) * np.sqrt(np.diag(sigma))

        l = len(Y)
        intervals = [[Y[k] - uncertainty[k], Y[k] + uncertainty[k]] for k in range(l)]

        return intervals

    def forecast_ahead_interval(self, data, steps, **kwargs):

        if 'alpha' in kwargs:
            alpha = kwargs.get('alpha')
        else:
            alpha = self.alpha

        smoothing = kwargs.get("smoothing", 0.5)

        X = [data[t] for t in np.arange(self.order+10)]
        S = []

        for k in np.arange(self.order, steps+self.order):
            sample = [[X[k-t-1] for t in np.arange(self.order)] for p in range(5)]
            Y, sigma = self.model_fit.predict([sample], return_std=True)
            X.append(Y[0])
            S.append(sigma[0])

        X = X[-steps:]

        intervals = []
        for k in range(steps):
            i = []
            i.append(X[k] - st.norm.ppf(alpha) * (1 + k*smoothing)*np.sqrt(S[k]))
            i.append(X[k] - st.norm.ppf(1-alpha) * (1 + k * smoothing) * np.sqrt(S[k]))
            intervals.append(i)

        return intervals

    def forecast_distribution(self, data, **kwargs):

        ret = []

        X = []
        l = len(data)
        for t in np.arange(self.order, l):
            X.append([data[t - k - 1] for k in np.arange(self.order)])

        Y, sigma = self.model_fit.predict(X, return_std=True)

        for k in len(Y):

            dist = ProbabilityDistribution.ProbabilityDistribution(type="histogram", uod=[self.original_min, self.original_max])
            intervals = []
            for alpha in np.arange(0.05, 0.5, 0.05):

                qt1 = Y[k] + st.norm.ppf(alpha) * sigma[k]
                qt2 = Y[k] + st.norm.ppf(1 - alpha) * sigma[k]

                intervals.append([qt1, qt2])

            dist.append_interval(intervals)

            ret.append(dist)

        return ret


    def forecast_ahead_distribution(self, data, steps, **kwargs):
        smoothing = kwargs.get("smoothing", 0.5)

        X = [data[t] for t in np.arange(self.order)]
        S = []

        for k in np.arange(self.order, steps+self.order):
            sample = [X[k-t-1] for t in np.arange(self.order)]
            Y, sigma = self.model_fit.predict([sample], return_std=True)
            X.append(Y[0])
            S.append(S[0])

        X = X[-steps:]
        #uncertainty = st.norm.ppf(alpha) * np.sqrt(np.diag(sigma))
        ret = []
        for k in range(steps):
            dist = ProbabilityDistribution.ProbabilityDistribution(type="histogram",
                                                                   uod=[self.original_min, self.original_max])
            intervals = []
            for alpha in np.arange(0.05, 0.5, 0.05):
                qt1 = X[k] - st.norm.ppf(alpha) * (1 + k*smoothing)*np.sqrt(S[k])
                qt2 = X[k] - st.norm.ppf(1-alpha) * (1 + k * smoothing) * np.sqrt(S[k])

                intervals.append([qt1, qt2])

            dist.append_interval(intervals)

            ret.append(dist)

        return ret








