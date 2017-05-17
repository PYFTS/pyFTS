#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tsa.tsatools import lagmat
from pyFTS import fts
from pyFTS.common import SortedCollection


class QuantileRegression(fts.FTS):
    """Façade for statsmodels.regression.quantile_regression"""
    def __init__(self, name, **kwargs):
        super(QuantileRegression, self).__init__(1, "")
        self.name = "QR"
        self.detail = "Quantile Regression"
        self.is_high_order = True
        self.has_point_forecasting = True
        self.has_interval_forecasting = True
        self.has_probability_forecasting = True
        self.has_probability_forecasting = True
        self.benchmark_only = True
        self.minOrder = 1
        self.alpha = kwargs.get("alpha", 0.05)
        self.dist = kwargs.get("dist", False)
        self.upper_qt = None
        self.mean_qt = None
        self.lower_qt = None
        self.dist_qt = None
        self.shortname = "QAR("+str(self.order)+","+str(self.alpha)+")"

    def train(self, data, sets, order=1, parameters=None):
        self.order = order

        tmp = np.array(self.doTransformations(data, updateUoD=True))

        lagdata, ndata = lagmat(tmp, maxlag=order, trim="both", original='sep')

        mqt = QuantReg(ndata, lagdata).fit(0.5)
        if self.alpha is not None:
            uqt = QuantReg(ndata, lagdata).fit(1 - self.alpha)
            lqt = QuantReg(ndata, lagdata).fit(self.alpha)

        self.mean_qt = [k for k in mqt.params]
        if self.alpha is not None:
            self.upper_qt = [k for k in uqt.params]
            self.lower_qt = [k for k in lqt.params]

        if self.dist:
            self.dist_qt = []
            for alpha in np.arange(0.05,0.5,0.05):
                lqt = QuantReg(ndata, lagdata).fit(alpha)
                uqt = QuantReg(ndata, lagdata).fit(1 - alpha)
                lo_qt = [k for k in lqt.params]
                up_qt = [k for k in uqt.params]
                self.dist_qt.append([lo_qt, up_qt])

        self.original_min = min(data)
        self.original_max = max(data)

        self.shortname = "QAR(" + str(self.order) + ") - " + str(self.alpha)

    def linearmodel(self,data,params):
        #return params[0] + sum([ data[k] * params[k+1] for k in np.arange(0, self.order) ])
        return sum([data[k] * params[k] for k in np.arange(0, self.order)])

    def point_to_interval(self, data, lo_params, up_params):
        lo = self.linearmodel(data, lo_params)
        up = self.linearmodel(data, up_params)
        return [lo, up]

    def interval_to_interval(self, data, lo_params, up_params):
        lo = self.linearmodel([k[0] for k in data], lo_params)
        up = self.linearmodel([k[1] for k in data], up_params)
        return [lo, up]

    def forecast(self, data, **kwargs):
        ndata = np.array(self.doTransformations(data))
        l = len(ndata)

        ret = []

        for k in np.arange(self.order, l+1):   #+1 to forecast one step ahead given all available lags
            sample = ndata[k - self.order : k]

            ret.append(self.linearmodel(sample, self.mean_qt))

        ret = self.doInverseTransformations(ret, params=[data[self.order - 1:]])

        return ret

    def forecastInterval(self, data, **kwargs):

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)

        ret = []

        for k in np.arange(self.order , l):
            sample = ndata[k - self.order: k]
            ret.append(self.point_to_interval(sample, self.lower_qt, self.upper_qt))

        ret = self.doInverseTransformations(ret, params=[data[self.order - 1:]], interval=True)

        return ret

    def forecastAheadInterval(self, data, steps, **kwargs):
        ndata = np.array(self.doTransformations(data))

        smoothing = kwargs.get("smoothing", 0.9)

        l = len(ndata)

        ret = []

        nmeans = self.forecastAhead(ndata, steps, **kwargs)

        for k in np.arange(0, self.order):
            nmeans.insert(k,ndata[-(k+1)])

        for k in np.arange(self.order, steps+self.order):
            intl = self.point_to_interval(nmeans[k - self.order: k], self.lower_qt, self.upper_qt)

            ret.append([intl[0]*(1 + k*smoothing), intl[1]*(1 + k*smoothing)])

        ret = self.doInverseTransformations(ret, params=[[data[-1] for a in np.arange(0, steps + self.order)]], interval=True)

        return ret[-steps:]

    def forecastAheadDistribution(self, data, steps, **kwargs):
        ndata = np.array(self.doTransformations(data))

        percentile_size = (self.original_max - self.original_min) / 100

        resolution = kwargs.get('resolution', percentile_size)

        grid = self.get_empty_grid(self.original_min, self.original_max, resolution)

        index = SortedCollection.SortedCollection(iterable=grid.keys())

        ret = []
        tmps = []

        grids = {}
        for k in np.arange(self.order, steps + self.order):
            grids[k] = self.get_empty_grid(self.original_min, self.original_max, resolution)

        for qt in self.dist_qt:
            intervals = [[k, k] for k in ndata[-self.order:]]
            for k in np.arange(self.order, steps + self.order):
                intl = self.interval_to_interval([intervals[x] for x in np.arange(k - self.order, k)], qt[0], qt[1])
                intervals.append(intl)
                grids[k] = self.gridCount(grids[k], resolution, index, intl)

        for k in np.arange(self.order, steps + self.order):
            tmp = np.array([grids[k][i] for i in sorted(grids[k])])
            ret.append(tmp / sum(tmp))

        grid = self.get_empty_grid(self.original_min, self.original_max, resolution)
        df = pd.DataFrame(ret, columns=sorted(grid))
        return df