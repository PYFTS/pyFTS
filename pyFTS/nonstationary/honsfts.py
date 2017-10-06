import numpy as np
from pyFTS.common import FuzzySet, FLR
from pyFTS import fts, hofts
from pyFTS.nonstationary import common, flrg


class HighOrderNonStationaryFLRG(flrg.NonStationaryFLRG):
    """First Order NonStationary Fuzzy Logical Relationship Group"""
    def __init__(self, order, **kwargs):
        super(HighOrderNonStationaryFLRG, self).__init__(order, **kwargs)

        self.LHS = []
        self.RHS = {}
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


class HighOrderNonStationaryFTS(hofts.HighOrderFLRG):
    """NonStationaryFTS Fuzzy Time Series"""
    def __init__(self, name, **kwargs):
        super(HighOrderNonStationaryFTS, self).__init__(1, "HONSFTS " + name, **kwargs)
        self.name = "High Order Non Stationary FTS"
        self.detail = ""
        self.flrgs = {}

    def generateFLRG(self, flrs):
        flrgs = {}
        l = len(flrs)
        for k in np.arange(self.order + 1, l):
            flrg = HighOrderNonStationaryFLRG(self.order)

            for kk in np.arange(k - self.order, k):
                flrg.appendLHS(flrs[kk].LHS)

            if flrg.strLHS() in flrgs:
                flrgs[flrg.strLHS()].appendRHS(flrs[k].RHS)
            else:
                flrgs[flrg.strLHS()] = flrg;
                flrgs[flrg.strLHS()].appendRHS(flrs[k].RHS)
        return (flrgs)

    def train(self, data, sets=None,order=1,parameters=None):

        if sets is not None:
            self.sets = sets
        else:
            self.sets = self.partitioner.sets

        ndata = self.doTransformations(data)
        tmpdata = common.fuzzySeries(ndata, self.sets)
        flrs = FLR.generateNonRecurrentFLRs(tmpdata)
        self.flrgs = self.generateFLRG(flrs)

    def forecast(self, data, **kwargs):

        time_displacement = kwargs.get("time_displacement",0)

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)

        ret = []

        for k in np.arange(0, l):

            #print("input: " + str(ndata[k]))

            tdisp = k + time_displacement

            affected_sets = [ [set, set.membership(ndata[k], tdisp)]
                              for set in self.sets if set.membership(ndata[k], tdisp) > 0.0]

            if len(affected_sets) == 0:
                if self.sets[0].get_lower(tdisp) > ndata[k]:
                    affected_sets.append([self.sets[0], 1.0])
                elif self.sets[-1].get_upper(tdisp) < ndata[k]:
                    affected_sets.append([self.sets[-1], 1.0])

            #print(affected_sets)

            tmp = []
            for aset in affected_sets:
                if aset[0] in self.flrgs:
                    tmp.append(self.flrgs[aset[0].name].get_midpoint(tdisp) * aset[1])
                else:
                    tmp.append(aset[0].get_midpoint(tdisp) * aset[1])

            pto = sum(tmp)

            #print(pto)

            ret.append(pto)

        ret = self.doInverseTransformations(ret, params=[data[self.order - 1:]])

        return ret

    def forecastInterval(self, data, **kwargs):

        time_displacement = kwargs.get("time_displacement",0)

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)

        ret = []

        for k in np.arange(0, l):

            tdisp = k + time_displacement

            affected_sets = [ [set.name, set.membership(ndata[k], tdisp)]
                              for set in self.sets if set.membership(ndata[k], tdisp) > 0.0]

            upper = []
            lower = []
            for aset in affected_sets:
                lower.append(self.flrgs[aset[0]].get_lower(tdisp) * aset[1])
                upper.append(self.flrgs[aset[0]].get_upper(tdisp) * aset[1])

            ret.append([sum(lower), sum(upper)])

        ret = self.doInverseTransformations(ret, params=[data[self.order - 1:]])

        return ret