import numpy as np
import pandas as pd
from pyFTS import tree
from pyFTS.common import FuzzySet, SortedCollection
from pyFTS.benchmarks import Measures


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
        self.isMultivariate = False
        self.dump = False
        self.transformations = []
        self.transformations_param = []
        self.original_max = 0
        self.original_min = 0
        self.partitioner = None

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

    def train(self, data, sets, order=1, parameters=None):
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

    def __len__(self):
       return len(self.flrgs)

    def len_total(self):
        return sum([len(k) for k in self.flrgs])

    def buildTreeWithoutOrder(self, node, lags, level):

        if level not in lags:
            return

        for s in lags[level]:
            node.appendChild(tree.FLRGTreeNode(s))

        for child in node.getChildren():
            self.buildTreeWithoutOrder(child, lags, level + 1)

    def inputoutputmapping(self,bins=100):

        dim_uod = tuple([bins for k in range(0,self.order)])

        dim_fs = tuple([ len(self.sets) for k in range(0, self.order)])

        simulation_uod = np.zeros(shape=dim_uod, dtype=float)

        simulation_fs = np.zeros(shape=dim_fs, dtype=float)

        percentiles = np.linspace(self.sets[0].lower, self.sets[-1].upper, bins).tolist()

        pdf_uod = {}

        for k in percentiles:
            pdf_uod[k] = 0

        pdf_fs = {}
        for k in self.sets:
            pdf_fs[k.name] = 0

        lags = {}

        for o in np.arange(0, self.order):
            lags[o] = percentiles

            # Build the tree with all possible paths

        root = tree.FLRGTreeNode(None)

        self.buildTreeWithoutOrder(root, lags, 0)

        # Trace the possible paths


        for p in root.paths():
            path = list(reversed(list(filter(None.__ne__, p))))

            index_uod = tuple([percentiles.index(k) for k in path])

            index_fs = tuple([ FuzzySet.getMaxMembershipFuzzySetIndex(k, self.sets) for k in path])

            forecast = self.forecast(path)[0]

            simulation_uod[index_uod] = forecast

            simulation_fs[index_fs] = forecast

        return [simulation_fs, simulation_uod ]





