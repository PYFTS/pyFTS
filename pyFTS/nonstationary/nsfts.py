import numpy as np
from pyFTS.common import FuzzySet, FLR
from pyFTS import fts, chen
from pyFTS.nonstationary import common, flrg


class ConventionalNonStationaryFLRG(flrg.NonStationaryFLRG):
    """First Order NonStationary Fuzzy Logical Relationship Group"""

    def __init__(self, LHS, **kwargs):
        super(ConventionalNonStationaryFLRG, self).__init__(1, **kwargs)
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


class NonStationaryFTS(fts.FTS):
    """NonStationaryFTS Fuzzy Time Series"""
    def __init__(self, name, **kwargs):
        super(NonStationaryFTS, self).__init__(1, "NSFTS " + name, **kwargs)
        self.name = "Non Stationary FTS"
        self.detail = ""
        self.flrgs = {}
        self.method = kwargs.get("method",'fuzzy')

    def generate_flrg(self, flrs, **kwargs):
        flrgs = {}
        for flr in flrs:
            if flr.LHS.name in flrgs:
                flrgs[flr.LHS.name].append(flr.RHS)
            else:
                flrgs[flr.LHS.name] = ConventionalNonStationaryFLRG(flr.LHS)
                flrgs[flr.LHS.name].append(flr.RHS)
        return flrgs

    def train(self, data, sets=None, order=1, parameters=None):

        if sets is not None:
            self.sets = sets
        else:
            self.sets = self.partitioner.sets

        ndata = self.doTransformations(data)
        window_size = parameters if parameters is not None else 1
        tmpdata = common.fuzzySeries(ndata, self.sets, window_size, method=self.method)
        #print([k[0].name for k in tmpdata])
        flrs = FLR.generateRecurrentFLRs(tmpdata)
        #print([str(k) for k in flrs])
        self.flrgs = self.generate_flrg(flrs)

    def forecast(self, data, **kwargs):

        time_displacement = kwargs.get("time_displacement",0)

        window_size = kwargs.get("window_size", 1)

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)

        ret = []

        for k in np.arange(0, l):

            #print("input: " + str(ndata[k]))

            tdisp = common.window_index(k + time_displacement, window_size)

            if self.method == 'fuzzy':
                affected_sets = [ [set, set.membership(ndata[k], tdisp)]
                                  for set in self.sets if set.membership(ndata[k], tdisp) > 0.0]
            elif self.method == 'maximum':
                mv = [set.membership(ndata[k], tdisp) for set in self.sets]
                ix = np.ravel(np.argwhere(mv == max(mv)))
                affected_sets = [self.sets[x] for x in ix]

            if len(affected_sets) == 0:
                if self.method == 'fuzzy':
                    affected_sets.append([common.check_bounds(ndata[k], self.sets, tdisp), 1.0])
                else:
                    affected_sets.append(common.check_bounds(ndata[k], self.sets, tdisp))

            #print(affected_sets)

            tmp = []

            if len(affected_sets) == 1 and self.method == 'fuzzy':
                aset = affected_sets[0][0]
                if aset.name in self.flrgs:
                    tmp.append(self.flrgs[aset.name].get_midpoint(tdisp))
                else:
                    tmp.append(aset.get_midpoint(tdisp))
            else:
                for aset in affected_sets:
                    if self.method == 'fuzzy':
                        if aset[0].name in self.flrgs:
                            tmp.append(self.flrgs[aset[0].name].get_midpoint(tdisp) * aset[1])
                        else:
                            tmp.append(aset[0].get_midpoint(tdisp) * aset[1])
                    elif self.method == 'maximum':
                        if aset.name in self.flrgs:
                            tmp.append(self.flrgs[aset.name].get_midpoint(tdisp))
                        else:
                            tmp.append(aset.get_midpoint(tdisp))

            pto = sum(tmp)

            #print(pto)

            ret.append(pto)

        ret = self.doInverseTransformations(ret, params=[data[self.order - 1:]])

        return ret

    def forecastInterval(self, data, **kwargs):

        time_displacement = kwargs.get("time_displacement", 0)

        window_size = kwargs.get("window_size", 1)

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)

        ret = []

        for k in np.arange(0, l):

            # print("input: " + str(ndata[k]))

            tdisp = common.window_index(k + time_displacement, window_size)

            if self.method == 'fuzzy':
                affected_sets = [[set, set.membership(ndata[k], tdisp)]
                                 for set in self.sets if set.membership(ndata[k], tdisp) > 0.0]
            elif self.method == 'maximum':
                mv = [set.membership(ndata[k], tdisp) for set in self.sets]
                ix = np.ravel(np.argwhere(mv == max(mv)))
                affected_sets = [self.sets[x] for x in ix]

            if len(affected_sets) == 0:
                if self.method == 'fuzzy':
                    affected_sets.append([common.check_bounds(ndata[k], self.sets, tdisp), 1.0])
                else:
                    affected_sets.append(common.check_bounds(ndata[k], self.sets, tdisp))

            upper = []
            lower = []

            if len(affected_sets) == 1:
                #print(2)
                aset = affected_sets[0][0]
                if aset.name in self.flrgs:
                    lower.append(self.flrgs[aset.name].get_lower(tdisp))
                    upper.append(self.flrgs[aset.name].get_upper(tdisp))
                else:
                    lower.append(aset.get_lower(tdisp))
                    upper.append(aset.get_upper(tdisp))
            else:
                for aset in affected_sets:
                    #print(aset)
                    if aset[0].name in self.flrgs:
                        lower.append(self.flrgs[aset[0].name].get_lower(tdisp) * aset[1])
                        upper.append(self.flrgs[aset[0].name].get_upper(tdisp) * aset[1])
                    else:
                        lower.append(aset[0].get_lower(tdisp) * aset[1])
                        upper.append(aset[0].get_upper(tdisp) * aset[1])


            ret.append([sum(lower), sum(upper)])

        ret = self.doInverseTransformations(ret, params=[data[self.order - 1:]])

        return ret