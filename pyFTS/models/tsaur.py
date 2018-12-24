"""
First order markov chain weighted FTS

Tsaur, Ruey-Chyn. A FUZZY TIME SERIES-MARKOV CHAIN MODEL WITH AN APPLICATION TO FORECAST THE EXCHANGE RATE
BETWEEN THE TAIWAN AND US DOLLAR. International Journal of Innovative Computing, Information and Control
vol 8, no 7(B), p. 4931–4942, 2012.
"""


import numpy as np
from pyFTS.common import FuzzySet,FLR,fts, flrg


class MarkovWeightedFLRG(flrg.FLRG):
    """First Order Markov Chain Weighted Fuzzy Logical Relationship Group"""
    def __init__(self, LHS, **kwargs):
        super(MarkovWeightedFLRG, self).__init__(1, **kwargs)
        self.LHS = LHS
        self.RHS = {}
        self.count = 0.0
        self.w = None

    def append_rhs(self, c, **kwargs):
        count = kwargs.get('count', 1.0)
        if c not in self.RHS:
            self.RHS[c] = count
        else:
            self.RHS[c] += count
        self.count += count

    def weights(self):
        if self.w is None:
            self.w = np.array([ v/self.count for k,v in self.RHS.items()])
        return self.w

    def get_midpoint(self, sets):
        mp = np.array([sets[c].centroid for c in self.RHS.keys()])
        return mp.dot(self.weights())

    def __str__(self):
        tmp = self.LHS + " -> "
        tmp2 = ""
        for c in sorted(self.RHS.keys()):
            if len(tmp2) > 0:
                tmp2 = tmp2 + ", "
            tmp2 = tmp2 + c + "(" + str(self.RHS[c]/self.count) + ")"
        return tmp + tmp2

    def __len__(self):
        return len(self.RHS)


class MarkovWeightedFTS(fts.FTS):
    """First Order Markov Chain Weighted Fuzzy Time Series"""
    def __init__(self, **kwargs):
        super(MarkovWeightedFTS, self).__init__(order=1, name="MWFTS", **kwargs)
        self.name = "Markov Weighted FTS"
        self.detail = "Tsaur"
        self.is_high_order = False
        self.order = 1

    def generate_flrg(self, flrs):
        for flr in flrs:
            if flr.LHS in self.flrgs:
                self.flrgs[flr.LHS].append_rhs(flr.RHS)
            else:
                self.flrgs[flr.LHS] = MarkovWeightedFLRG(flr.LHS)
                self.flrgs[flr.LHS].append_rhs(flr.RHS)

    def train(self, data, **kwargs):
        tmpdata = FuzzySet.fuzzyfy(data, partitioner=self.partitioner, method='maximum', mode='sets')
        flrs = FLR.generate_recurrent_flrs(tmpdata)
        self.generate_flrg(flrs)

    def forecast(self, ndata, **kwargs):

        explain = kwargs.get('explain', False)

        if self.partitioner is not None:
            ordered_sets = self.partitioner.ordered_sets
        else:
            ordered_sets = FuzzySet.set_ordered(self.sets)

        data = np.array(ndata)

        l = len(ndata)

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
