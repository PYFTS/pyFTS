import numpy as np
from pyFTS.common import FuzzySet, FLR
from pyFTS import fts, hofts
from pyFTS.nonstationary import common, flrg
from pyFTS import tree


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


class HighOrderNonStationaryFTS(hofts.HighOrderFTS):
    """NonStationaryFTS Fuzzy Time Series"""
    def __init__(self, name, **kwargs):
        super(HighOrderNonStationaryFTS, self).__init__("HONSFTS " + name, **kwargs)
        self.name = "High Order Non Stationary FTS"
        self.detail = ""
        self.flrgs = {}

    def generate_flrg(self, data, **kwargs):
        flrgs = {}
        l = len(data)
        window_size = kwargs.get("window_size", 1)
        for k in np.arange(self.order, l):
            if self.dump: print("FLR: " + str(k))

            sample = data[k - self.order: k]

            disp = common.window_index(k, window_size)

            rhs = [set for set in self.sets if set.membership(data[k], disp) > 0.0]

            if len(rhs) == 0:
                rhs = [common.check_bounds(data[k], self.sets, disp)]

            lags = {}

            for o in np.arange(0, self.order):
                tdisp = common.window_index(k - (self.order - o), window_size)
                lhs = [set for set in self.sets if set.membership(sample[o], tdisp) > 0.0]

                if len(lhs) == 0:
                    lhs = [common.check_bounds(sample[o], self.sets, tdisp)]

                lags[o] = lhs

            root = tree.FLRGTreeNode(None)

            self.build_tree_without_order(root, lags, 0)

            # Trace the possible paths
            for p in root.paths():
                flrg = HighOrderNonStationaryFLRG(self.order)
                path = list(reversed(list(filter(None.__ne__, p))))

                for c, e in enumerate(path, start=0):
                    flrg.appendLHS(e)

                if flrg.strLHS() not in flrgs:
                    flrgs[flrg.strLHS()] = flrg;

                for st in rhs:
                    flrgs[flrg.strLHS()].appendRHS(st)

        return flrgs

    def train(self, data, sets=None, order=2, parameters=None):

        self.order = order

        if sets is not None:
            self.sets = sets
        else:
            self.sets = self.partitioner.sets

        ndata = self.doTransformations(data)
        #tmpdata = common.fuzzySeries(ndata, self.sets)
        #flrs = FLR.generateRecurrentFLRs(ndata)
        window_size = parameters if parameters is not None else 1
        self.flrgs = self.generate_flrg(ndata, window_size=window_size)

    def forecast(self, data, **kwargs):

        time_displacement = kwargs.get("time_displacement",0)

        window_size = kwargs.get("window_size", 1)

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)

        ret = []

        for k in np.arange(self.order, l+1):

            #print("input: " + str(ndata[k]))

            disp = common.window_index(k + time_displacement, window_size)

            affected_flrgs = []
            affected_flrgs_memberships = []

            lags = {}

            sample = ndata[k - self.order: k]

            for ct, dat in enumerate(sample):
                tdisp = common.window_index((k + time_displacement) - (self.order - ct), window_size)
                sel = [ct for ct, set in enumerate(self.sets) if set.membership(dat, tdisp) > 0.0]

                if len(sel) == 0:
                    sel.append(common.check_bounds_index(dat, self.sets, tdisp))

                lags[ct] = sel

            # Build the tree with all possible paths

            root = tree.FLRGTreeNode(None)

            self.build_tree(root, lags, 0)

            # Trace the possible paths and build the PFLRG's

            for p in root.paths():
                path = list(reversed(list(filter(None.__ne__, p))))
                flrg = HighOrderNonStationaryFLRG(self.order)

                for kk in path:
                    flrg.appendLHS(self.sets[kk])

                affected_flrgs.append(flrg)
                affected_flrgs_memberships.append(flrg.get_membership(ndata[k - self.order: k], disp))

            #print(affected_sets)

            tmp = []
            for ct, aset in enumerate(affected_flrgs):
                if aset.strLHS() in self.flrgs:
                    tmp.append(self.flrgs[aset.strLHS()].get_midpoint(tdisp) *
                               affected_flrgs_memberships[ct])
                else:
                    tmp.append(aset.LHS[-1].get_midpoint(tdisp))

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