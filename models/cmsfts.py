import numpy as np
from pyFTS.common import FuzzySet,FLR
from pyFTS import fts, sfts, chen


class ContextualSeasonalFLRG(object):
    def __init__(self, seasonality):
        super(ContextualSeasonalFLRG, self).__init__(None,None)
        self.season = seasonality
        self.flrgs = {}

    def append(self, flr):
        if flr.LHS.name in self.flrgs:
            self.flrgs[flr.LHS.name].append(flr.RHS)
        else:
            self.flrgs[flr.LHS.name] = chen.ConventionalFLRG(flr.LHS)
            self.flrgs[flr.LHS.name].append(flr.RHS)

    def __str__(self):
        tmp = str(self.season) + ": \n "
        tmp2 = "\t"
        for r in sorted(self.flrgs):
            tmp2 += str(self.flrgs[r]) + "\n\t"
        return tmp + tmp2 + "\n"


class ContextualMultiSeasonalFTS(sfts.SeasonalFTS):
    def __init__(self, name, indexer):
        super(ContextualMultiSeasonalFTS, self).__init__("CMSFTS")
        self.name = "Contextual Multi Seasonal FTS"
        self.shortname = "CMSFTS " + name
        self.detail = ""
        self.seasonality = 1
        self.hasSeasonality = True
        self.hasPointForecasting = True
        self.isHighOrder = True
        self.isMultivariate = True
        self.indexer = indexer
        self.flrgs = {}

    def generateFLRG(self, flrs):
        flrgs = {}

        for flr in flrs:

            if str(flr.index) not in self.flrgs:
                flrgs[str(flr.index)] = ContextualSeasonalFLRG(flr.index)

            flrgs[str(flr.index)].append(flr)

        return (flrgs)

    def train(self, data, sets, order=1, parameters=None):
        self.sets = sets
        self.seasonality = parameters
        flrs = FLR.generateIndexedFLRs(self.sets, self.indexer, data)
        self.flrgs = self.generateFLRG(flrs)

    def getMidpoints(self, flrg, data):
        ret = np.array([s.centroid for s in flrg.flrgs[data].RHS])
        return ret

    def forecast(self, data):

        ret = []

        index = self.indexer.get_season_of_data(data)
        ndata = self.indexer.get_data(data)

        for k in np.arange(1, len(data)):

            flrg = self.flrgs[str(index[k])]

            d = FuzzySet.getMaxMembershipFuzzySet(ndata[k], self.sets)

            mp = self.getMidpoints(flrg, d)

            ret.append(sum(mp) / len(mp))

        ret = self.doInverseTransformations(ret, params=[ndata[self.order - 1:]])

        return ret

    def forecastAhead(self, data, steps):
        ret = []
        for i in steps:
            flrg = self.flrgs[str(i)]

            mp = self.getMidpoints(flrg)

            ret.append(sum(mp) / len(mp))

        ret = self.doInverseTransformations(ret, params=data)

        return ret
