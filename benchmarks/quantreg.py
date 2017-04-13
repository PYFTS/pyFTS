#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
from pyFTS import fts


class QuantileRegression(fts.FTS):
    def __init__(self, name):
        super(QuantileRegression, self).__init__(1, "QR")
        self.name = "QR"
        self.detail = "Quantile Regression"
        self.isHighOrder = True
        self.hasIntervalForecasting = True
        self.benchmark_only = True
        self.minOrder = 1
        self.alpha = 0.5

    def train(self, data, sets, order=1, parameters=None):
        pass

    def forecast(self, data):
        pass
