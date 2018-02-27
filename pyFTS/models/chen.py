"""
First Order Conventional Fuzzy Time Series by Chen (1996)

S.-M. Chen, “Forecasting enrollments based on fuzzy time series,” Fuzzy Sets Syst., vol. 81, no. 3, pp. 311–319, 1996.
"""

import numpy as np
from pyFTS.common import FuzzySet, FLR, fts, flrg


class ConventionalFLRG(flrg.FLRG):
    """First Order Conventional Fuzzy Logical Relationship Group"""
    def __init__(self, LHS, **kwargs):
        super(ConventionalFLRG, self).__init__(1, **kwargs)
        self.LHS = LHS
        self.RHS = set()

    def append(self, c):
        self.RHS.add(c)

    def __str__(self):
        tmp = self.LHS.name + " -> "
        tmp2 = ""
        for c in sorted(self.RHS, key=lambda s: s.name):
            if len(tmp2) > 0:
                tmp2 = tmp2 + ","
            tmp2 = tmp2 + c.name
        return tmp + tmp2


class ConventionalFTS(fts.FTS):
    """Conventional Fuzzy Time Series"""
    def __init__(self, name, **kwargs):
        super(ConventionalFTS, self).__init__(1, "CFTS " + name, **kwargs)
        self.name = "Conventional FTS"
        self.detail = "Chen"
        self.flrgs = {}

    def generateFLRG(self, flrs):
        flrgs = {}
        for flr in flrs:
            if flr.LHS.name in flrgs:
                flrgs[flr.LHS.name].append(flr.RHS)
            else:
                flrgs[flr.LHS.name] = ConventionalFLRG(flr.LHS)
                flrgs[flr.LHS.name].append(flr.RHS)
        return (flrgs)

    def train(self, data, sets,order=1,parameters=None):
        self.sets = sets
        ndata = self.apply_transformations(data)
        tmpdata = FuzzySet.fuzzyfy_series_old(ndata, sets)
        flrs = FLR.generate_non_recurrent_flrs(tmpdata)
        self.flrgs = self.generateFLRG(flrs)

    def forecast(self, data, **kwargs):

        ndata = np.array(self.apply_transformations(data))

        l = len(ndata)

        ret = []

        for k in np.arange(0, l):

            mv = FuzzySet.fuzzyfy_instance(ndata[k], self.sets)

            actual = self.sets[np.argwhere(mv == max(mv))[0, 0]]

            if actual.name not in self.flrgs:
                ret.append(actual.centroid)
            else:
                _flrg = self.flrgs[actual.name]

                ret.append(_flrg.get_midpoint())

        ret = self.apply_inverse_transformations(ret, params=[data[self.order - 1:]])

        return ret
