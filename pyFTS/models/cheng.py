"""
Trend Weighted Fuzzy Time Series by Cheng, Chen and Wu (2009)

C.-H. Cheng, Y.-S. Chen, and Y.-L. Wu, “Forecasting innovation diffusion of products using trend-weighted fuzzy time-series model,” 
Expert Syst. Appl., vol. 36, no. 2, pp. 1826–1832, 2009.
"""

import numpy as np
from pyFTS.common import FuzzySet, FLR, fts
from pyFTS.models import yu


class TrendWeightedFLRG(yu.WeightedFLRG):
    """
    First Order Trend Weighted Fuzzy Logical Relationship Group
    """
    def __init__(self, LHS, **kwargs):
        super(TrendWeightedFLRG, self).__init__(LHS, **kwargs)
        self.w = None

    def weights(self, sets):
        if self.w is None:
            count_nochange = 0.0
            count_up = 0.0
            count_down = 0.0
            weights = []

            for c in self.RHS:
                tmp = 0
                if sets[self.LHS].centroid == sets[c].centroid:
                    count_nochange += 1.0
                    tmp = count_nochange
                elif sets[self.LHS].centroid > sets[c].centroid:
                    count_down += 1.0
                    tmp = count_down
                else:
                    count_up += 1.0
                    tmp = count_up
                weights.append(tmp)

            tot = sum(weights)
            self.w = np.array([k / tot for k in weights])
        return self.w


class TrendWeightedFTS(yu.WeightedFTS):
    """First Order Trend Weighted Fuzzy Time Series"""
    def __init__(self, name, **kwargs):
        super(TrendWeightedFTS, self).__init__("", **kwargs)
        self.shortname = "TWFTS " + name
        self.name = "Trend Weighted FTS"
        self.detail = "Cheng"
        self.is_high_order = False

    def generate_FLRG(self, flrs):
        for flr in flrs:
            if flr.LHS in self.flrgs:
                self.flrgs[flr.LHS].append_rhs(flr.RHS)
            else:
                self.flrgs[flr.LHS] = TrendWeightedFLRG(flr.LHS)
                self.flrgs[flr.LHS].append_rhs(flr.RHS)
