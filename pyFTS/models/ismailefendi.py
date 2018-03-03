"""
First Order Improved Weighted Fuzzy Time Series by Efendi, Ismail and Deris (2013)

R. Efendi, Z. Ismail, and M. M. Deris, “Improved weight Fuzzy Time Series as used in the exchange rates forecasting of 
US Dollar to Ringgit Malaysia,” Int. J. Comput. Intell. Appl., vol. 12, no. 1, p. 1350005, 2013.
"""

import numpy as np
from pyFTS.common import FuzzySet, FLR, fts, flrg


class ImprovedWeightedFLRG(flrg.FLRG):
    """First Order Improved Weighted Fuzzy Logical Relationship Group"""
    def __init__(self, LHS, **kwargs):
        super(ImprovedWeightedFLRG, self).__init__(1, **kwargs)
        self.LHS = LHS
        self.RHS = {}
        self.rhs_counts = {}
        self.count = 0.0
        self.w = None

    def append_rhs(self, c, **kwargs):
        if c not in self.RHS:
            self.RHS[c] = c
            self.rhs_counts[c] = 1.0
        else:
            self.rhs_counts[c] += 1.0
        self.count += 1.0

    def weights(self):
        if self.w is None:
            self.w = np.array([self.rhs_counts[c] / self.count for c in self.RHS.keys()])
        return self.w

    def __str__(self):
        tmp = self.LHS + " -> "
        tmp2 = ""
        for c in sorted(self.RHS.keys()):
            if len(tmp2) > 0:
                tmp2 = tmp2 + ","
            tmp2 = tmp2 + c + "(" + str(round(self.rhs_counts[c] / self.count, 3)) + ")"
        return tmp + tmp2

    def __len__(self):
        return len(self.RHS)


class ImprovedWeightedFTS(fts.FTS):
    """First Order Improved Weighted Fuzzy Time Series"""
    def __init__(self, name, **kwargs):
        super(ImprovedWeightedFTS, self).__init__(1, "IWFTS " + name, **kwargs)
        self.name = "Improved Weighted FTS"
        self.detail = "Ismail & Efendi"

    def generate_flrg(self, flrs):
        for flr in flrs:
            if flr.LHS in self.flrgs:
                self.flrgs[flr.LHS].append_rhs(flr.RHS)
            else:
                self.flrgs[flr.LHS] = ImprovedWeightedFLRG(flr.LHS);
                self.flrgs[flr.LHS].append_rhs(flr.RHS)

    def train(self, data, **kwargs):
        if kwargs.get('sets', None) is not None:
            self.sets = kwargs.get('sets', None)

        ndata = self.apply_transformations(data)

        tmpdata = FuzzySet.fuzzyfy_series(ndata, self.sets, method="maximum")
        flrs = FLR.generate_recurrent_flrs(tmpdata)
        self.generate_flrg(flrs)

    def forecast(self, data, **kwargs):
        l = 1

        ordered_sets = FuzzySet.set_ordered(self.sets)

        data = np.array(data)
        ndata = self.apply_transformations(data)

        l = len(ndata)

        ret = []

        for k in np.arange(0, l):

            actual = FuzzySet.get_maximum_membership_fuzzyset(ndata[k], self.sets, ordered_sets)

            if actual.name not in self.flrgs:
                ret.append(actual.centroid)
            else:
                flrg = self.flrgs[actual.name]
                mp = flrg.get_midpoints(self.sets)

                ret.append(mp.dot(flrg.weights()))

        ret = self.apply_inverse_transformations(ret, params=[data])

        return ret
