import numpy as np
from pyFTS.common import FuzzySet,FLR
from pyFTS import fts


class WeightedFLRG(fts.FTS):
    def __init__(self, LHS):
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
    def __init__(self, name):
        super(WeightedFTS, self).__init__(1, "WFTS")
        self.name = "Weighted FTS"
        self.detail = "Yu"

    def generateFLRG(self, flrs):
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
        ndata = self.doTransformations(data)
        tmpdata = FuzzySet.fuzzySeries(ndata, sets)
        flrs = FLR.generateRecurrentFLRs(tmpdata)
        self.flrgs = self.generateFLRG(flrs)

    def forecast(self, data):
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
