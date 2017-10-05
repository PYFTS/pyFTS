import numpy as np
from pyFTS.common import FuzzySet, FLR
from pyFTS import fts, chen
from pyFTS.nonstationary import common


class NonStationaryFLRG(chen.ConventionalFLRG):
    """First Order NonStationary Fuzzy Logical Relationship Group"""
    def __init__(self, LHS):
        super(NonStationaryFLRG, self).__init__(LHS)

    def get_midpoint(self, t):
        if self.midpoint is None:
            tmp = []
            for r in self.RHS:
                tmp.append(r.get_midpoint(t))
            self.midpoint = sum(tmp)/len(tmp)
        return self.midpoint

    def get_lower(self, t):
        if self.lower is None:
            tmp = []
            for r in self.RHS:
                tmp.append(r.get_midpoint(t))
            self.lower = min(tmp)
        return self.lower

    def get_upper(self, t):
        if self.upper is None:
            tmp = []
            for r in self.RHS:
                tmp.append(r.get_midpoint(t))
            self.upper = max(tmp)
        return self.upper


class NonStationaryFTS(fts.FTS):
    """NonStationaryFTS Fuzzy Time Series"""
    def __init__(self, name, **kwargs):
        super(NonStationaryFTS, self).__init__(1, "NSFTS " + name, **kwargs)
        self.name = "Non Stationary FTS"
        self.detail = ""
        self.flrgs = {}

    def generateFLRG(self, flrs):
        flrgs = {}
        for flr in flrs:
            if flr.LHS.name in flrgs:
                flrgs[flr.LHS.name].append(flr.RHS)
            else:
                flrgs[flr.LHS.name] = NonStationaryFLRG(flr.LHS)
                flrgs[flr.LHS.name].append(flr.RHS)
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

            tdisp = k + time_displacement

            affected_sets = [ [set, set.membership(ndata[k], tdisp)]
                              for set in self.sets if set.membership(ndata[k], tdisp) > 0.0]

            tmp = []
            for aset in affected_sets:
                if aset[0] in self.flrgs:
                    tmp.append(self.flrgs[aset[0].name].get_midpoint(tdisp) * aset[1])
                else:
                    tmp.append(aset[0].get_midpoint(tdisp) * aset[1])

            ret.append(sum(tmp))

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