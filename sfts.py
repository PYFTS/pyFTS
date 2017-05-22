"""
Simple First Order Seasonal Fuzzy Time Series implementation of Song (1999) based of Conventional FTS by Chen (1996)

Q. Song, “Seasonal forecasting in fuzzy time series,” Fuzzy sets Syst., vol. 107, pp. 235–236, 1999.

S.-M. Chen, “Forecasting enrollments based on fuzzy time series,” Fuzzy Sets Syst., vol. 81, no. 3, pp. 311–319, 1996.
"""

import numpy as np
from pyFTS.common import FuzzySet,FLR
from pyFTS import fts


class SeasonalFLRG(FLR.FLR):
    """First Order Seasonal Fuzzy Logical Relationship Group"""
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
    """First Order Seasonal Fuzzy Time Series"""
    def __init__(self, name, **kwargs):
        super(SeasonalFTS, self).__init__(1, "SFTS")
        self.name = "Seasonal FTS"
        self.detail = "Chen"
        self.seasonality = 1
        self.has_seasonality = True
        self.has_point_forecasting = True
        self.is_high_order = False

    def generateFLRG(self, flrs):
        flrgs = {}
        for ct, flr in enumerate(flrs, start=1):

            season = self.indexer.get_season_by_index(ct)[0]

            ss = str(season)

            if ss not in flrgs:
                flrgs[ss] = SeasonalFLRG(season)

            #print(season)
            flrgs[ss].append(flr.RHS)

        return (flrgs)

    def train(self, data, sets, order=1, parameters=None):
        self.sets = sets
        ndata = self.doTransformations(data)
        tmpdata = FuzzySet.fuzzySeries(ndata, sets)
        flrs = FLR.generateRecurrentFLRs(tmpdata)
        self.flrgs = self.generateFLRG(flrs)

    def forecast(self, data, **kwargs):

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)

        ret = []

        for k in np.arange(1, l):

            season = self.indexer.get_season_by_index(k)[0]

            flrg = self.flrgs[str(season)]

            mp = self.getMidpoints(flrg)

            ret.append(np.percentile(mp, 50))

        ret = self.doInverseTransformations(ret, params=[data[self.order - 1:]])

        return ret
