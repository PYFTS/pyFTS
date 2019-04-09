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
        self.w = None

    def append_rhs(self, c, **kwargs):
        count = kwargs.get('count', 1.0)
        self.RHS.append(c)
        self.count += count

    def weights(self, sets):
        if self.w is None:
            tot = sum(np.arange(1.0, self.count, 1.0))
            self.w = np.array([k / tot for k in np.arange(1.0, self.count, 1.0)])
        return self.w

    def __str__(self):
        tmp = self.LHS + " -> "
        tmp2 = ""
        cc = 1.0
        tot = sum(np.arange(1.0, self.count, 1.0))
        for c in sorted(self.RHS):
            if len(tmp2) > 0:
                tmp2 = tmp2 + ","
            tmp2 = tmp2 + c + "(" + str(round(cc / tot, 3)) + ")"
            cc = cc + 1.0
        return tmp + tmp2


class WeightedFTS(fts.FTS):
    """First Order Weighted Fuzzy Time Series"""
    def __init__(self, **kwargs):
        super(WeightedFTS, self).__init__(order=1, name="WFTS", **kwargs)
        self.name = "Weighted FTS"
        self.detail = "Yu"

    def generate_FLRG(self, flrs):
        for flr in flrs:
            if flr.LHS in self.flrgs:
                self.flrgs[flr.LHS].append_rhs(flr.RHS)
            else:
                self.flrgs[flr.LHS] = WeightedFLRG(flr.LHS);
                self.flrgs[flr.LHS].append_rhs(flr.RHS)

    def train(self, ndata, **kwargs):
        tmpdata = self.partitioner.fuzzyfy(ndata, method='maximum', mode='sets')
        flrs = FLR.generate_recurrent_flrs(tmpdata)
        self.generate_FLRG(flrs)

    def forecast(self, ndata, **kwargs):

        explain = kwargs.get('explain', False)

        if self.partitioner is not None:
            ordered_sets = self.partitioner.ordered_sets
        else:
            ordered_sets = FuzzySet.set_ordered(self.sets)

        ndata = np.array(ndata)

        l = len(ndata) if not explain else 1

        ret = []

        for k in np.arange(0, l):

            actual = FuzzySet.get_maximum_membership_fuzzyset(ndata[k], self.sets, ordered_sets)

            if explain:
                print("Fuzzyfication:\n\n {} -> {} \n\n".format(ndata[k], actual.name))

            if actual.name not in self.flrgs:
                ret.append(actual.centroid)

                if explain:
                    print("Rules:\n\n {} -> {} (Naïve)\t Midpoint: {}  \n\n".format(actual.name, actual.name,actual.centroid))

            else:
                flrg = self.flrgs[actual.name]
                mp = flrg.get_midpoints(self.sets)

                final = mp.dot(flrg.weights(self.sets))

                ret.append(final)

                if explain:
                    print("Rules:\n\n {} \n\n ".format(str(flrg)))
                    print("Midpoints: \n\n {}\n\n".format(mp))

                    print("Deffuzyfied value: {} \n".format(final))

        return ret
