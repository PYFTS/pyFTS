import numpy as np
from pyFTS.common import FuzzySet, FLR
from pyFTS.models.seasonal import sfts
from pyFTS.models import chen


class ContextualSeasonalFLRG(sfts.SeasonalFLRG):
    """
    Contextual Seasonal Fuzzy Logical Relationship Group
    """
    def __init__(self, seasonality):
        super(ContextualSeasonalFLRG, self).__init__(seasonality)
        self.RHS = {}

    def append_rhs(self, flr, **kwargs):
        if flr.LHS in self.RHS:
            self.RHS[flr.LHS].append_rhs(flr.RHS)
        else:
            self.RHS[flr.LHS] = chen.ConventionalFLRG(flr.LHS)
            self.RHS[flr.LHS].append_rhs(flr.RHS)

    def __str__(self):
        tmp = str(self.LHS) + ": \n "
        tmp2 = "\t"
        for r in sorted(self.RHS):
            tmp2 += str(self.RHS[r]) + "\n\t"
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

            self.flrgs[str(flr.index)].append_rhs(flr)

    def train(self, data,  **kwargs):
        if kwargs.get('sets', None) is not None:
            self.sets = kwargs.get('sets', None)
        if kwargs.get('parameters', None) is not None:
            self.seasonality = kwargs.get('parameters', None)
        flrs = FLR.generate_indexed_flrs(self.sets, self.indexer, data)
        self.generate_flrg(flrs)

    def get_midpoints(self, flrg, data):
        if data in flrg.flrgs:
            ret = np.array([self.sets[s].centroid for s in flrg.flrgs[data.name].RHS])
            return ret
        else:
            return  np.array([self.sets[data].centroid])

    def forecast(self, data, **kwargs):
        ordered_sets = FuzzySet.set_ordered(self.sets)

        ret = []

        index = self.indexer.get_season_of_data(data)
        ndata = self.indexer.get_data(data)

        for k in np.arange(0, len(data)):

            flrg = self.flrgs[str(index[k])]

            d = FuzzySet.get_maximum_membership_fuzzyset(ndata[k], self.sets, ordered_sets)

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
