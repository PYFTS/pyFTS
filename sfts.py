import numpy as np
from pyFTS.common import FuzzySet,FLR
import fts


class SeasonalFLRG(fts.FTS):
    def __init__(self, seasonality):
        self.LHS = seasonality
        self.RHS = []

    def append(self, c):
        self.RHS.append(c)

    def __str__(self):
        tmp = str(self.LHS) + " -> "
        tmp2 = ""
        for c in sorted(self.RHS, key=lambda s: s.name):
            if len(tmp2) > 0:
                tmp2 = tmp2 + ","
            tmp2 = tmp2 + c.name
        return tmp + tmp2


class SeasonalFTS(fts.FTS):
    def __init__(self, name):
        super(SeasonalFTS, self).__init__(1, "SFTS")
        self.name = "Seasonal FTS"
        self.detail = "Chen"
        self.seasonality = 1
        self.isSeasonal = True

    def generateFLRG(self, flrs):
        flrgs = []
        season = 1
        for flr in flrs:
            if len(flrgs) < self.seasonality:
                flrgs.append(SeasonalFLRG(season))

            flrgs[season].append(flr.RHS)

            season = (season + 1) % (self.seasonality + 1)

            if season == 0: season = 1

        return (flrgs)

    def train(self, data, sets, seasonality):
        self.sets = sets
        self.seasonality = seasonality
        tmpdata = FuzzySet.fuzzySeries(data, sets)
        flrs = FLR.generateRecurrentFLRs(tmpdata)
        self.flrgs = self.generateFLRG(flrs)

    def forecast(self, data):

        ndata = np.array(data)

        l = len(ndata)

        ret = []

        for k in np.arange(1, l):
            flrg = self.flrgs[data[k]]

            mp = self.getMidpoints(flrg)

            ret.append(sum(mp) / len(mp))

        return ret
