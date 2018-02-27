"""
First Order Weighted Fuzzy Time Series by Yu(2005)

H.-K. Yu, “Weighted fuzzy time series models for TAIEX forecasting,” 
Phys. A Stat. Mech. its Appl., vol. 349, no. 3, pp. 609–624, 2005.
"""

import numpy as np
from pyFTS.common import FuzzySet, FLR, fts, flrg
from pyFTS.models import chen


class WeightedFLRG(flrg.FLRG):
    """First Order Weighted Fuzzy Logical Relationship Group"""
    def __init__(self, LHS, **kwargs):
        super(WeightedFLRG, self).__init__(1, **kwargs)
        self.LHS = LHS
        self.RHS = []
        self.count = 1.0

    def append(self, c):
        self.RHS.append(c)
        self.count = self.count + 1.0

    def weights(self):
        tot = sum(np.arange(1.0, self.count, 1.0))
        return np.array([k / tot for k in np.arange(1.0, self.count, 1.0)])

    def __str__(self):
        tmp = self.LHS.name + " -> "
        tmp2 = ""
        cc = 1.0
        tot = sum(np.arange(1.0, self.count, 1.0))
        for c in sorted(self.RHS, key=lambda s: s.name):
            if len(tmp2) > 0:
                tmp2 = tmp2 + ","
            tmp2 = tmp2 + c.name + "(" + str(round(cc / tot, 3)) + ")"
            cc = cc + 1.0
        return tmp + tmp2


class WeightedFTS(fts.FTS):
    """First Order Weighted Fuzzy Time Series"""
    def __init__(self, name, **kwargs):
        super(WeightedFTS, self).__init__(1, "WFTS " + name, **kwargs)
        self.name = "Weighted FTS"
        self.detail = "Yu"

    def generate_FLRG(self, flrs):
        flrgs = {}
        for flr in flrs:
            if flr.LHS.name in flrgs:
                flrgs[flr.LHS.name].append(flr.RHS)
            else:
                flrgs[flr.LHS.name] = WeightedFLRG(flr.LHS);
                flrgs[flr.LHS.name].append(flr.RHS)
        return (flrgs)

    def train(self, data, sets,order=1,parameters=None):
        self.sets = sets
        ndata = self.apply_transformations(data)
        tmpdata = FuzzySet.fuzzyfy_series_old(ndata, sets)
        flrs = FLR.generate_recurrent_flrs(tmpdata)
        self.flrgs = self.generate_FLRG(flrs)

    def forecast(self, data, **kwargs):
        l = 1

        data = np.array(data)

        ndata = self.apply_transformations(data)

        l = len(ndata)

        ret = []

        for k in np.arange(0, l):

            mv = FuzzySet.fuzzyfy_instance(ndata[k], self.sets)

            actual = self.sets[np.argwhere(mv == max(mv))[0, 0]]

            if actual.name not in self.flrgs:
                ret.append(actual.centroid)
            else:
                flrg = self.flrgs[actual.name]
                mp = flrg.get_midpoints()

                ret.append(mp.dot(flrg.weights()))

        ret = self.apply_inverse_transformations(ret, params=[data[self.order - 1:]])

        return ret
