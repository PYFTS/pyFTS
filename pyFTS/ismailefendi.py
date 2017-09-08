"""
First Order Improved Weighted Fuzzy Time Series by Efendi, Ismail and Deris (2013)

R. Efendi, Z. Ismail, and M. M. Deris, “Improved weight Fuzzy Time Series as used in the exchange rates forecasting of 
US Dollar to Ringgit Malaysia,” Int. J. Comput. Intell. Appl., vol. 12, no. 1, p. 1350005, 2013.
"""

import numpy as np
from pyFTS.common import FuzzySet,FLR
from pyFTS import fts


class ImprovedWeightedFLRG(object):
    """First Order Improved Weighted Fuzzy Logical Relationship Group"""
    def __init__(self, LHS):
        self.LHS = LHS
        self.RHS = {}
        self.count = 0.0

    def append(self, c):
        if c.name not in self.RHS:
            self.RHS[c.name] = 1.0
        else:
            self.RHS[c.name] = self.RHS[c.name] + 1.0
        self.count = self.count + 1.0

    def weights(self):
        return np.array([self.RHS[c] / self.count for c in self.RHS.keys()])

    def __str__(self):
        tmp = self.LHS.name + " -> "
        tmp2 = ""
        for c in sorted(self.RHS):
            if len(tmp2) > 0:
                tmp2 = tmp2 + ","
            tmp2 = tmp2 + c + "(" + str(round(self.RHS[c] / self.count, 3)) + ")"
        return tmp + tmp2

    def __len__(self):
        return len(self.RHS)


class ImprovedWeightedFTS(fts.FTS):
    """First Order Improved Weighted Fuzzy Time Series"""
    def __init__(self, name, **kwargs):
        super(ImprovedWeightedFTS, self).__init__(1, "IWFTS " + name, **kwargs)
        self.name = "Improved Weighted FTS"
        self.detail = "Ismail & Efendi"
        self.setsDict = {}

    def generateFLRG(self, flrs):
        flrgs = {}
        for flr in flrs:
            if flr.LHS.name in flrgs:
                flrgs[flr.LHS.name].append(flr.RHS)
            else:
                flrgs[flr.LHS.name] = ImprovedWeightedFLRG(flr.LHS);
                flrgs[flr.LHS.name].append(flr.RHS)
        return (flrgs)

    def train(self, data, sets, order=1, parameters=None):
        self.sets = sets

        for s in self.sets:    self.setsDict[s.name] = s

        ndata = self.doTransformations(data)

        tmpdata = FuzzySet.fuzzySeries(ndata, self.sets)
        flrs = FLR.generateRecurrentFLRs(tmpdata)
        self.flrgs = self.generateFLRG(flrs)

    def getMidpoints(self, flrg):
        ret = np.array([self.setsDict[s].centroid for s in flrg.RHS])
        return ret

    def forecast(self, data, **kwargs):
        l = 1

        data = np.array(data)
        ndata = self.doTransformations(data)

        l = len(ndata)

        ret = []

        for k in np.arange(0, l):

            mv = FuzzySet.fuzzyInstance(ndata[k], self.sets)

            actual = self.sets[np.argwhere(mv == max(mv))[0, 0]]

            if actual.name not in self.flrgs:
                ret.append(actual.centroid)
            else:
                flrg = self.flrgs[actual.name]
                mp = self.getMidpoints(flrg)

                ret.append(mp.dot(flrg.weights()))

        ret = self.doInverseTransformations(ret, params=[data[self.order - 1:]])

        return ret
