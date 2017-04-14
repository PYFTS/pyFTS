#!/usr/bin/python
# -*- coding: utf8 -*-

from pyFTS import fts


class Naive(fts.FTS):
    def __init__(self, order, name, **kwargs):
        super(Naive, self).__init__(1, "Naive " + name)
        self.name = "Naïve Model"
        self.detail = "Naïve Model"
        self.benchmark_only = True
        self.isHighOrder = False

    def forecast(self, data, **kwargs):
        return [k for k in data]

