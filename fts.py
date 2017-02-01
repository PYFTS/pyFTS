import numpy as np
from pyFTS import *


class FTS(object):
    def __init__(self, order, name):
        self.sets = {}
        self.flrgs = {}
        self.order = order
        self.shortname = name
        self.name = name
        self.detail = name
        self.isHighOrder = False
        self.minOrder = 1
        self.hasSeasonality = False
        self.hasPointForecasting = True
        self.hasIntervalForecasting = False
        self.hasDistributionForecasting = False
        self.dump = False
        self.transformations = []
        self.transformations_param = []
        self.original_max = 0
        self.original_min = 0

    def fuzzy(self, data):
        best = {"fuzzyset": "", "membership": 0.0}

        for f in self.sets:
            fset = self.sets[f]
            if best["membership"] <= fset.membership(data):
                best["fuzzyset"] = fset.name
                best["membership"] = fset.membership(data)

        return best

    def forecast(self, data):
        pass

    def forecastInterval(self, data):
        pass

    def forecastDistribution(self, data):
        pass

    def forecastAhead(self, data, steps):
        pass

    def forecastAheadInterval(self, data, steps):
        pass

    def forecastAheadDistribution(self, data, steps):
        pass

    def train(self, data, sets,order=1, parameters=None):
        pass

    def getMidpoints(self, flrg):
        ret = np.array([s.centroid for s in flrg.RHS])
        return ret

    def appendTransformation(self, transformation):
        self.transformations.append(transformation)

    def doTransformations(self,data,params=None,updateUoD=False):
        ndata = data
        if updateUoD:
            if min(data) < 0:
                self.original_min = min(data) * 1.1
            else:
                self.original_min = min(data) * 0.9

            if max(data) > 0:
                self.original_max = max(data) * 1.1
            else:
                self.original_max = max(data) * 0.9

        if len(self.transformations) > 0:
            if params is None:
                params = [ None for k in self.transformations]

            for c, t in enumerate(self.transformations, start=0):
                ndata = t.apply(ndata,params[c])

        return ndata

    def doInverseTransformations(self, data, params=None):
        ndata = data
        if len(self.transformations) > 0:
            if params is None:
                params = [None for k in self.transformations]

            for c, t in enumerate(reversed(self.transformations), start=0):
                ndata = t.inverse(ndata, params[c])

        return ndata

    def __str__(self):
        tmp = self.name + ":\n"
        for r in sorted(self.flrgs):
            tmp = tmp + str(self.flrgs[r]) + "\n"
        return tmp
