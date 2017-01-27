import numpy as np
from pyFTS.common import FuzzySet,FLR
from pyFTS import fts


class HighOrderFLRG:
    def __init__(self, order):
        self.LHS = []
        self.RHS = {}
        self.order = order
        self.strlhs = ""

    def appendRHS(self, c):
        if c.name not in self.RHS:
            self.RHS[c.name] = c

    def strLHS(self):
        if len(self.strlhs) == 0:
            for c in self.LHS:
                if len(self.strlhs) > 0:
                    self.strlhs += ", "
                self.strlhs = self.strlhs + c.name
        return self.strlhs

    def appendLHS(self, c):
        self.LHS.append(c)

    def __str__(self):
        tmp = ""
        for c in sorted(self.RHS):
            if len(tmp) > 0:
                tmp = tmp + ","
            tmp = tmp + c
        return self.strLHS() + " -> " + tmp


class HighOrderFTS(fts.FTS):
    def __init__(self, name):
        super(HighOrderFTS, self).__init__(1, "HOFTS" + name)
        self.name = "High Order FTS"
        self.shortname = "HOFTS" + name
        self.detail = "Chen"
        self.order = 1
        self.setsDict = {}
        self.isHighOrder = True

    def generateFLRG(self, flrs):
        flrgs = {}
        l = len(flrs)
        for k in np.arange(self.order + 1, l):
            flrg = HighOrderFLRG(self.order)

            for kk in np.arange(k - self.order, k):
                flrg.appendLHS(flrs[kk].LHS)

            if flrg.strLHS() in flrgs:
                flrgs[flrg.strLHS()].appendRHS(flrs[k].RHS)
            else:
                flrgs[flrg.strLHS()] = flrg;
                flrgs[flrg.strLHS()].appendRHS(flrs[k].RHS)
        return (flrgs)

    def train(self, data, sets, order=1,parameters=None):

        data = self.doTransformations(data)

        self.order = order
        self.sets = sets
        for s in self.sets:    self.setsDict[s.name] = s
        tmpdata = FuzzySet.fuzzySeries(data, sets)
        flrs = FLR.generateRecurrentFLRs(tmpdata)
        self.flrgs = self.generateFLRG(flrs)

    def getMidpoints(self, flrg):
        ret = np.array([self.setsDict[s].centroid for s in flrg.RHS])
        return ret

    def forecast(self, data):

        ret = []

        l = len(data)

        if l <= self.order:
            return data

        ndata = self.doTransformations(data)

        for k in np.arange(self.order, l+1):
            tmpdata = FuzzySet.fuzzySeries(ndata[k - self.order: k], self.sets)
            tmpflrg = HighOrderFLRG(self.order)

            for s in tmpdata: tmpflrg.appendLHS(s)

            if tmpflrg.strLHS() not in self.flrgs:
                ret.append(tmpdata[-1].centroid)
            else:
                flrg = self.flrgs[tmpflrg.strLHS()]
                mp = self.getMidpoints(flrg)

                ret.append(sum(mp) / len(mp))

        ret = self.doInverseTransformations(ret, params=[data[self.order-1:]])

        return ret
