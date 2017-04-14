#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tsa.tsatools import lagmat
from pyFTS import fts


class QuantileRegression(fts.FTS):
    def __init__(self, order, **kwargs):
        super(QuantileRegression, self).__init__(1, "QR")
        self.name = "QR"
        self.detail = "Quantile Regression"
        self.isHighOrder = True
        self.hasPointForecasting = True
        self.hasIntervalForecasting = True
        self.benchmark_only = True
        self.minOrder = 1
        self.alpha = 0.5
        self.upper_qt = None
        self.mean_qt = None
        self.lower_qt = None

    def train(self, data, sets, order=1, parameters=None):
        self.order = order

        if parameters is not None:
            self.alpha = parameters

        tmp = np.array(self.doTransformations(data))

        lagdata, ndata = lagmat(tmp, maxlag=order, trim="both", original='sep')
        uqt = QuantReg(ndata, lagdata).fit(1 - self.alpha)
        mqt = QuantReg(ndata, lagdata).fit(0.5)
        lqt = QuantReg(ndata, lagdata).fit(self.alpha)

        self.upper_qt = [uqt.params[k] for k in uqt.params.keys()]
        self.mean_qt = [mqt.params[k] for k in mqt.params.keys()]
        self.lower_qt = [lqt.params[k] for k in lqt.params.keys()]

    def linearmodel(self,data,params):
        return params[0] + sum([ data[k] * params[k+1] for k in np.arange(0, self.order) ])

    def forecast(self, data, **kwargs):
        ndata = np.array(self.doTransformations(data))
        l = len(ndata)

        ret = []

        for k in np.arange(self.order, l):
            sample = ndata[k - self.order : k]

            ret.append(self.linearmodel(sample, self.mean_qt))

        ret = self.doInverseTransformations(ret, params=[data[self.order - 1:]])

        return ret

    def forecastInterval(self, data, **kwargs):

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)

        ret = []

        for k in np.arange(self.order - 1, l):
            sample = ndata[k - self.order: k]
            up = self.linearmodel(sample, self.upper_qt)
            down = self.linearmodel(sample, self.down_qt)
            ret.append([up, down])

        ret = self.doInverseTransformations(ret, params=[data[self.order - 1:]])

        return ret
