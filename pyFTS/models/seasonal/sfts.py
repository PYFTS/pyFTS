"""
Simple First Order Seasonal Fuzzy Time Series implementation of Song (1999) based of Conventional FTS by Chen (1996)

Q. Song, “Seasonal forecasting in fuzzy time series,” Fuzzy sets Syst., vol. 107, pp. 235–236, 1999.

S.-M. Chen, “Forecasting enrollments based on fuzzy time series,” Fuzzy Sets Syst., vol. 81, no. 3, pp. 311–319, 1996.
"""

import numpy as np
from pyFTS.common import FuzzySet, FLR, flrg, fts


class SeasonalFLRG(flrg.FLRG):
    """First Order Seasonal Fuzzy Logical Relationship Group"""
    def __init__(self, seasonality):
        super(SeasonalFLRG, self).__init__(1)
        self.LHS = seasonality
        self.RHS = []

    def get_key(self):
        return self.LHS

    def append_rhs(self, c, **kwargs):
        self.RHS.append(c)

    def __str__(self):
        tmp = str(self.LHS) + " -> "
        tmp2 = ""
        for c in sorted(self.RHS, key=lambda s: str(s)):
            if len(tmp2) > 0:
                tmp2 = tmp2 + ","
            tmp2 = tmp2 + str(c)
        return tmp + tmp2

    def __len__(self):
        return len(self.RHS)


class SeasonalFTS(fts.FTS):
    """First Order Seasonal Fuzzy Time Series"""
    def __init__(self, **kwargs):
        super(SeasonalFTS, self).__init__(**kwargs)
        self.name = "Seasonal FTS"
        self.shortname = "SFTS"
        self.order = 1
        self.seasonality = 1
        self.has_seasonality = True
        self.has_point_forecasting = True
        self.is_high_order = False
        self.flrgs = {}

    def generate_flrg(self, flrs):

        for ct, flr in enumerate(flrs, start=1):

            season = self.indexer.get_season_by_index(ct)[0]

            ss = str(season)

            if ss not in self.flrgs:
                self.flrgs[ss] = SeasonalFLRG(season)

            #print(season)
            self.flrgs[ss].append_rhs(flr.RHS)

    def get_midpoints(self, flrg):
        ret = np.array([self.sets[s].centroid for s in flrg.RHS])
        return ret

    def train(self, data,  **kwargs):
        if kwargs.get('sets', None) is not None:
            self.sets = kwargs.get('sets', None)
        tmpdata = FuzzySet.fuzzyfy_series_old(data, self.sets)
        flrs = FLR.generate_non_recurrent_flrs(tmpdata)
        self.generate_flrg(flrs)

    def forecast(self, data, **kwargs):

        l = len(data)

        ret = []

        for k in np.arange(0, l):

            season = self.indexer.get_season_by_index(k)[0]

            flrg = self.flrgs[str(season)]

            mp = self.get_midpoints(flrg)

            ret.append(np.percentile(mp, 50))

        return ret

    def __str__(self):
        """String representation of the model"""

        tmp = self.name + ":\n"
        for r in self.flrgs:
            tmp = tmp + str(self.flrgs[r]) + "\n"
        return tmp
