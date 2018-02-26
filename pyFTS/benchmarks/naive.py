#!/usr/bin/python
# -*- coding: utf8 -*-

from pyFTS.common import fts


class Naive(fts.FTS):
    """Naïve Forecasting method"""
    def __init__(self, name, **kwargs):
        super(Naive, self).__init__(1, "Naive")
        self.name = "Naïve Model"
        self.detail = "Naïve Model"
        self.benchmark_only = True
        self.is_high_order = False

    def forecast(self, data, **kwargs):
        return data

