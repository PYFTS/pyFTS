#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
import math
from operator import itemgetter
from pyFTS.common import FLR, FuzzySet, SortedCollection
from pyFTS import fts


class EnsembleFTS(fts.FTS):
    def __init__(self, name, **kwargs):
        super(EnsembleFTS, self).__init__("Ensemble FTS")
        self.shortname = "Ensemble FTS " + name
        self.name = "Ensemble FTS"
        self.flrgs = {}
        self.has_point_forecasting = True
        self.has_interval_forecasting = True
        self.has_probability_forecasting = True
        self.is_high_order = True
        self.models = []
        self.parameters = []

    def build(self, data, models, partitioners, partitions, max_order=3, transformation=None, indexer=None):

        self.models = []

        for count, model in enumerate(models, start=0):
            mfts = model("")
            if mfts.benchmark_only:
                if transformation is not None:
                    mfts.appendTransformation(transformation)
                mfts.train(data,None, order=1, parameters=None)
                self.models.append(mfts)
            else:
                for partition in partitions:
                    for partitioner in partitioners:
                        data_train_fs = partitioner(data, partition, transformation=transformation)
                        mfts = model("")

                        mfts.partitioner = data_train_fs
                        if not mfts.is_high_order:

                            if transformation is not None:
                                mfts.appendTransformation(transformation)

                            mfts.train(data, data_train_fs.sets)
                            self.models.append(mfts)
                        else:
                            for order in np.arange(1, max_order + 1):
                                if order >= mfts.min_order:
                                    mfts = model("")
                                    mfts.partitioner = data_train_fs

                                    if transformation is not None:
                                        mfts.appendTransformation(transformation)

                                    mfts.train(data, data_train_fs.sets, order=order)
                                    self.models.append(mfts)

    def train(self, data, sets, order=1,parameters=None):
        pass

    def forecast(self, data, **kwargs):

        method = kwargs.get('method','mean')

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)

        ret = []

        for k in np.arange(0, l+1):
            tmp = []
            for model in self.models:
                if k >= model.minOrder - 1:
                    sample = ndata[k - model.order : k]
                    tmp.append( model.forecast(sample) )
            if method == 'mean':
                ret.append( np.nanmean(tmp))
            elif method == 'median':
                ret.append(np.percentile(tmp,50))

        ret = self.doInverseTransformations(ret, params=[data[self.order - 1:]])

        return ret

    def forecastInterval(self, data, **kwargs):

        method = kwargs.get('method', 'extremum')

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)

        ret = []

        for k in np.arange(0, l):
            tmp = []
            for model in self.models:
                if k >= model.minOrder - 1:
                    sample = ndata[k - model.order : k]
                    tmp.append( model.forecast(sample) )
            if method == 'extremum':
                ret.append( [ min(tmp), max(tmp) ] )
            elif method == 'quantile':
                q = kwargs.get('q', [.05, .95])
                ret.append(np.percentile(tmp,q=q*100))

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
        #print(point_to_interval)
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

