import numpy as np
from pyFTS.common import FuzzySet,FLR
from pyFTS import fts, sfts

class MultiSeasonalFTS(sfts.SeasonalFTS):
    def __init__(self, name, indexer):
        super(MultiSeasonalFTS, self).__init__("MSFTS")
        self.name = "Multi Seasonal FTS"
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

        for index, season in enumerate(self.indexer.get_season_of_data(flrs),start=0):

            print(index)
            print(season)

            if str(season) not in self.flrgs:
                flrgs[str(season)] = sfts.SeasonalFLRG(season)

            flrgs[str(season)].append(flrs[index].RHS)

        return (flrgs)

    def train(self, data, sets, order=1, parameters=None):
        self.sets = sets
        self.seasonality = parameters
        ndata = self.indexer.set_data(data,self.doTransformations(self.indexer.get_data(data)))
        tmpdata = FuzzySet.fuzzySeries(ndata, sets)
        flrs = FLR.generateRecurrentFLRs(tmpdata)
        self.flrgs = self.generateFLRG(flrs)

    def forecast(self, data):

        ndata = np.array(self.doTransformations(self.indexer.get_data(data)))

        l = len(ndata)

        ret = []

        for k in np.arange(1, l):

            season = self.indexer.get_season_index(k)

            flrg = self.flrgs[str(season)]

            mp = self.getMidpoints(flrg)

            ret.append(sum(mp) / len(mp))

        ret = self.doInverseTransformations(ret, params=[data[self.order - 1:]])

        return ret
