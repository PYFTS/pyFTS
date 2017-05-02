import numpy as np
from pyFTS.common import FuzzySet,FLR
from pyFTS import fts, sfts


class MultiSeasonalFTS(sfts.SeasonalFTS):
    """
    Multi-Seasonal Fuzzy Time Series
    """
    def __init__(self, order, name, indexer, **kwargs):
        super(MultiSeasonalFTS, self).__init__("MSFTS")
        self.name = "Multi Seasonal FTS"
        self.shortname = "MSFTS " + name
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
                flrgs[str(flr.index)] = sfts.SeasonalFLRG(flr.index)

            flrgs[str(flr.index)].append(flr.RHS)

        return (flrgs)

    def train(self, data, sets, order=1, parameters=None):
        self.sets = sets
        self.seasonality = parameters
        #ndata = self.indexer.set_data(data,self.doTransformations(self.indexer.get_data(data)))
        flrs = FLR.generateIndexedFLRs(self.sets, self.indexer, data)
        self.flrgs = self.generateFLRG(flrs)

    def forecast(self, data, **kwargs):

        ret = []

        index = self.indexer.get_season_of_data(data)
        ndata = self.indexer.get_data(data)

        for k in np.arange(1, len(data)):

            flrg = self.flrgs[str(index[k])]

            mp = self.getMidpoints(flrg)

            ret.append(sum(mp) / len(mp))

        ret = self.doInverseTransformations(ret, params=[ndata[self.order - 1:]])

        return ret

    def forecastAhead(self, data, steps, **kwargs):
        ret = []
        for i in steps:
            flrg = self.flrgs[str(i)]

            mp = self.getMidpoints(flrg)

            ret.append(sum(mp) / len(mp))

        ret = self.doInverseTransformations(ret, params=data)

        return ret
