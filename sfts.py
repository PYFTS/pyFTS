import numpy as np
from pyFTS.common import FuzzySet,FLR
from pyFTS import fts


class SeasonalFLRG(FLR.FLR):
    def __init__(self, seasonality):
        super(SeasonalFLRG, self).__init__(None,None)
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

    def __len__(self):
        return len(self.RHS)


class SeasonalFTS(fts.FTS):
    def __init__(self, order, **kwargs):
        super(SeasonalFTS, self).__init__(1, "SFTS")
        self.name = "Seasonal FTS"
        self.detail = "Chen"
        self.seasonality = 1
        self.hasSeasonality = True
        self.hasPointForecasting = True
        self.isHighOrder = False

    def generateFLRG(self, flrs):
        flrgs = []
        season = 1
        for flr in flrs:

            if len(flrgs) < self.seasonality:
                flrgs.append(SeasonalFLRG(season))

            #print(season)
            flrgs[season-1].append(flr.RHS)

            season = (season + 1) % (self.seasonality + 1)

            if season == 0: season = 1

        return (flrgs)

    def train(self, data, sets, order=1,parameters=12):
        self.sets = sets
        self.seasonality = parameters
        ndata = self.doTransformations(data)
        tmpdata = FuzzySet.fuzzySeries(ndata, sets)
        flrs = FLR.generateRecurrentFLRs(tmpdata)
        self.flrgs = self.generateFLRG(flrs)

    def forecast(self, data, **kwargs):

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)

        ret = []

        for k in np.arange(1, l):
            #flrg = self.flrgs[ndata[k]]

            season = (k + 1) % (self.seasonality + 1)

            #print(season)

            flrg = self.flrgs[season-1]

            mp = self.getMidpoints(flrg)

            ret.append(sum(mp) / len(mp))

        ret = self.doInverseTransformations(ret, params=[data[self.order - 1:]])

        return ret
