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
        count = kwargs.get('count', 1.0)
        if c not in self.RHS:
            self.RHS[c] = c
            self.rhs_counts[c] = count
        else:
            self.rhs_counts[c] += count
        self.count += count

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
    def __init__(self, **kwargs):
        super(ImprovedWeightedFTS, self).__init__(order=1, name="IWFTS", **kwargs)
        self.name = "Improved Weighted FTS"
        self.detail = "Ismail & Efendi"

    def generate_flrg(self, flrs):
        for flr in flrs:
            if flr.LHS in self.flrgs:
                self.flrgs[flr.LHS].append_rhs(flr.RHS)
            else:
                self.flrgs[flr.LHS] = ImprovedWeightedFLRG(flr.LHS)
                self.flrgs[flr.LHS].append_rhs(flr.RHS)

    def train(self, ndata, **kwargs):

        tmpdata = FuzzySet.fuzzyfy(ndata, partitioner=self.partitioner, method='maximum', mode='sets')
        flrs = FLR.generate_recurrent_flrs(tmpdata)
        self.generate_flrg(flrs)

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
                print("Fuzzyfication:\n\n {} -> {} \n".format(ndata[k], actual.name))

            if actual.name not in self.flrgs:
                ret.append(actual.centroid)

                if explain:
                    print("Rules:\n\n {} -> {} (Naïve)\t Midpoint: {}  \n\n".format(actual.name, actual.name,actual.centroid))

            else:
                flrg = self.flrgs[actual.name]
                mp = flrg.get_midpoints(self.sets)

                final = mp.dot(flrg.weights())

                ret.append(final)

                if explain:
                    print("Rules:\n\n {} \n\n ".format(str(flrg)))
                    print("Midpoints: \n\n {}\n\n".format(mp))

                    print("Deffuzyfied value: {} \n".format(final))

        return ret
