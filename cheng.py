import numpy as np
from pyFTS.common import FuzzySet,FLR
from pyFTS import fts, yu


class TrendWeightedFLRG(yu.WeightedFTS):
    """First Order Trend Weighted Fuzzy Logical Relationship Group"""
    def __init__(self, LHS, **kwargs):
        super(TrendWeightedFTS, self).__init__(LHS)

    def weights(self):
        count_nochange = 0.0
        count_up = 0.0
        count_down = 0.0
        weights = []

        for c in self.RHS:
            tmp = 0
            if self.RHS.midpoint == c.midpoint:
                count_nochange += 1.0
                tmp = count_nochange
            elif self.RHS.midpoint > c.midpoint:
                count_down += 1.0
                tmp = count_down
            else:
                count_up += 1.0
                tmp = count_up
            weights.append(tmp)

        tot = sum(weights)
        return np.array([k / tot for k in weights])


class TrendWeightedFTS(yu.WeightedFTS):
    """First Order Trend Weighted Fuzzy Time Series"""
    def __init__(self, name, **kwargs):
        super(TrendWeightedFTS, self).__init__(1, "TWFTS " + name)
        self.name = "Trend Weighted FTS"
        self.detail = "Cheng"

    def generateFLRG(self, flrs):
        flrgs = {}
        for flr in flrs:
            if flr.LHS.name in flrgs:
                flrgs[flr.LHS.name].append(flr.RHS)
            else:
                flrgs[flr.LHS.name] = TrendWeightedFLRG(flr.LHS);
                flrgs[flr.LHS.name].append(flr.RHS)
        return (flrgs)