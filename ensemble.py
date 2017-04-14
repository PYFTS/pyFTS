#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
import math
from operator import itemgetter
from pyFTS.common import FLR, FuzzySet, SortedCollection
from pyFTS import fts

class EnsembleFTS(fts.FTS):
    def __init__(self, order, name, **kwargs):
        super(EnsembleFTS, self).__init__("Ensemble FTS")
        self.shortname = "Ensemble FTS " + name
        self.name = "Ensemble FTS"
        self.flrgs = {}
        self.hasPointForecasting = True
        self.hasIntervalForecasting = True
        self.hasDistributionForecasting = True
        self.isHighOrder = True
        self.models = []
        self.parameters = []

    def train(self, data, sets, order=1,parameters=None):

        pass

    def forecast(self, data, **kwargs):

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)

        ret = []

        for k in np.arange(self.order - 1, l):
            pass

        ret = self.doInverseTransformations(ret, params=[data[self.order - 1:]])

        return ret

    def forecastInterval(self, data, **kwargs):

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)

        ret = []

        for k in np.arange(self.order - 1, l):
            pass

        return ret

    def forecastAhead(self, data, steps, **kwargs):
        pass

    def forecastAheadInterval(self, data, steps, **kwargs):
        pass


    def getGridClean(self, resolution):
        grid = {}

        if len(self.transformations) == 0:
            _min = self.sets[0].lower
            _max = self.sets[-1].upper
        else:
            _min = self.original_min
            _max = self.original_max

        for sbin in np.arange(_min,_max, resolution):
            grid[sbin] = 0

        return grid

    def gridCount(self, grid, resolution, index, interval):
        #print(interval)
        for k in index.inside(interval[0],interval[1]):
            #print(k)
            grid[k] += 1
        return grid

    def gridCountPoint(self, grid, resolution, index, point):
        k = index.find_ge(point)
        # print(k)
        grid[k] += 1
        return grid

    def forecastAheadDistribution(self, data, steps, **kwargs):
        pass

