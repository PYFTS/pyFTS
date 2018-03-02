import numpy as np
from pyFTS.common import FuzzySet, FLR
from pyFTS.models.seasonal import sfts
from pyFTS.models import chen


class ContextualSeasonalFLRG(object):
    """
    Contextual Seasonal Fuzzy Logical Relationship Group
    """
    def __init__(self, seasonality):
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
    """
    Contextual Multi-Seasonal Fuzzy Time Series
    """
    def __init__(self, name, indexer, **kwargs):
        super(ContextualMultiSeasonalFTS, self).__init__("CMSFTS")
        self.name = "Contextual Multi Seasonal FTS"
        self.shortname = "CMSFTS " + name
        self.detail = ""
        self.seasonality = 1
        self.has_seasonality = True
        self.has_point_forecasting = True
        self.is_high_order = True
        self.is_multivariate = True
        self.indexer = indexer
        self.flrgs = {}

    def generate_flrg(self, flrs):
        for flr in flrs:

            if str(flr.index) not in self.flrgs:
                self.flrgs[str(flr.index)] = ContextualSeasonalFLRG(flr.index)

            self.flrgs[str(flr.index)].append(flr)

    def train(self, data,  **kwargs):
        if kwargs.get('sets', None) is not None:
            self.sets = kwargs.get('sets', None)
        if kwargs.get('parameters', None) is not None:
            self.seasonality = kwargs.get('parameters', None)
        flrs = FLR.generate_indexed_flrs(self.sets, self.indexer, data)
        self.generate_flrg(flrs)

    def get_midpoints(self, flrg, data):
        if data.name in flrg.flrgs:
            ret = np.array([s.centroid for s in flrg.flrgs[data.name].RHS])
            return ret
        else:
            return  np.array([data.centroid])

    def forecast(self, data, **kwargs):

        ret = []

        index = self.indexer.get_season_of_data(data)
        ndata = self.indexer.get_data(data)

        for k in np.arange(0, len(data)):

            flrg = self.flrgs[str(index[k])]

            d = FuzzySet.get_maximum_membership_fuzzyset(ndata[k], self.sets)

            mp = self.get_midpoints(flrg, d)

            ret.append(sum(mp) / len(mp))

        ret = self.doInverseTransformations(ret, params=[ndata])

        return ret

    def forecast_ahead(self, data, steps, **kwargs):
        ret = []
        for i in steps:
            flrg = self.flrgs[str(i)]

            mp = self.get_midpoints(flrg)

            ret.append(sum(mp) / len(mp))

        ret = self.doInverseTransformations(ret, params=data)

        return ret
