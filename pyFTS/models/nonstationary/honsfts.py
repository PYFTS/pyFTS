import numpy as np
from pyFTS.common import FuzzySet, FLR, fts, tree
from pyFTS.models import hofts
from pyFTS.nonstationary import common, flrg


class HighOrderNonStationaryFLRG(flrg.NonStationaryFLRG):
    """First Order NonStationary Fuzzy Logical Relationship Group"""
    def __init__(self, order, **kwargs):
        super(HighOrderNonStationaryFLRG, self).__init__(order, **kwargs)

        self.LHS = []
        self.RHS = {}

    def append_rhs(self, c, **kwargs):
        if c.name not in self.RHS:
            self.RHS[c.name] = c

    def append_lhs(self, c):
        self.LHS.append(c)

    def __str__(self):
        tmp = ""
        for c in sorted(self.RHS):
            if len(tmp) > 0:
                tmp = tmp + ","
            tmp = tmp + c
        return self.get_key() + " -> " + tmp


class HighOrderNonStationaryFTS(hofts.HighOrderFTS):
    """NonStationaryFTS Fuzzy Time Series"""
    def __init__(self, name, **kwargs):
        super(HighOrderNonStationaryFTS, self).__init__("HONSFTS " + name, **kwargs)
        self.name = "High Order Non Stationary FTS"
        self.detail = ""
        self.flrgs = {}

    def generate_flrg(self, data, **kwargs):
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
                    flrg.append_lhs(e)

                if flrg.get_key() not in self.flrgs:
                    self.flrgs[flrg.get_key()] = flrg;

                for st in rhs:
                    self.flrgs[flrg.get_key()].append_rhs(st)

        # flrgs = sorted(flrgs, key=lambda flrg: flrg.get_midpoint(0, window_size=1))

    def train(self, data, **kwargs):

        if kwargs.get('order', None) is not None:
            self.order = kwargs.get('order', 1)

        if kwargs.get('sets', None) is not None:
            self.sets = kwargs.get('sets', None)

        ndata = self.apply_transformations(data)
        #tmpdata = common.fuzzyfy_series_old(ndata, self.sets)
        #flrs = FLR.generate_recurrent_flrs(ndata)
        window_size = kwargs.get('parameters', 1)
        self.generate_flrg(ndata, window_size=window_size)

    def _affected_flrgs(self, sample, k, time_displacement, window_size):
        # print("input: " + str(ndata[k]))

        affected_flrgs = []
        affected_flrgs_memberships = []

        lags = {}

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
                flrg.append_lhs(self.sets[kk])

            affected_flrgs.append(flrg)
            # affected_flrgs_memberships.append_rhs(flrg.get_membership(sample, disp))

            #                print(flrg.get_key())

            # the FLRG is here because of the bounds verification
            mv = []
            for ct, dat in enumerate(sample):
                td = common.window_index((k + time_displacement) - (self.order - ct), window_size)
                tmp = flrg.LHS[ct].membership(dat, td)
                # print('td',td)
                # print('dat',dat)
                # print(flrg.LHS[ct].name, flrg.LHS[ct].perturbated_parameters[td])
                # print(tmp)

                if (tmp == 0.0 and flrg.LHS[ct].name == self.sets[0].name and dat < self.sets[0].get_lower(td)) \
                        or (tmp == 0.0 and flrg.LHS[ct].name == self.sets[-1].name and dat > self.sets[-1].get_upper(
                            td)):
                    mv.append(1.0)
                else:
                    mv.append(tmp)
            # print(mv)

            affected_flrgs_memberships.append(np.prod(mv))

        return [affected_flrgs, affected_flrgs_memberships]

    def forecast(self, data, **kwargs):

        time_displacement = kwargs.get("time_displacement",0)

        window_size = kwargs.get("window_size", 1)

        ndata = np.array(self.apply_transformations(data))

        l = len(ndata)

        ret = []

        for k in np.arange(self.order, l+1):

            sample = ndata[k - self.order: k]

            affected_flrgs, affected_flrgs_memberships = self._affected_flrgs(sample, k,
                                                                              time_displacement, window_size)

            #print([str(k) for k in affected_flrgs])
            #print(affected_flrgs_memberships)

            tmp = []
            tdisp = common.window_index(k + time_displacement, window_size)
            if len(affected_flrgs) == 0:
                tmp.append(common.check_bounds(sample[-1], self.sets, tdisp))
            elif len(affected_flrgs) == 1:
                flrg = affected_flrgs[0]
                if flrg.get_key() in self.flrgs:
                    tmp.append(self.flrgs[flrg.get_key()].get_midpoint(tdisp))
                else:
                    tmp.append(flrg.LHS[-1].get_midpoint(tdisp))
            else:
                for ct, aset in enumerate(affected_flrgs):
                    if aset.get_key() in self.flrgs:
                        tmp.append(self.flrgs[aset.get_key()].get_midpoint(tdisp) *
                                   affected_flrgs_memberships[ct])
                    else:
                        tmp.append(aset.LHS[-1].get_midpoint(tdisp)*
                                   affected_flrgs_memberships[ct])
            pto = sum(tmp)

            #print(pto)

            ret.append(pto)

        ret = self.apply_inverse_transformations(ret, params=[data[self.order - 1:]])

        return ret

    def forecast_interval(self, data, **kwargs):

        time_displacement = kwargs.get("time_displacement", 0)

        window_size = kwargs.get("window_size", 1)

        ndata = np.array(self.apply_transformations(data))

        l = len(ndata)

        ret = []

        for k in np.arange(self.order, l + 1):

            sample = ndata[k - self.order: k]

            affected_flrgs, affected_flrgs_memberships = self._affected_flrgs(sample, k,
                                                                              time_displacement, window_size)

            # print([str(k) for k in affected_flrgs])
            # print(affected_flrgs_memberships)

            upper = []
            lower = []

            tdisp = common.window_index(k + time_displacement, window_size)
            if len(affected_flrgs) == 0:
                aset = common.check_bounds(sample[-1], self.sets, tdisp)
                lower.append(aset.get_lower(tdisp))
                upper.append(aset.get_upper(tdisp))
            elif len(affected_flrgs) == 1:
                _flrg = affected_flrgs[0]
                if _flrg.get_key() in self.flrgs:
                    lower.append(self.flrgs[_flrg.get_key()].get_lower(tdisp))
                    upper.append(self.flrgs[_flrg.get_key()].get_upper(tdisp))
                else:
                    lower.append(_flrg.LHS[-1].get_lower(tdisp))
                    upper.append(_flrg.LHS[-1].get_upper(tdisp))
            else:
                for ct, aset in enumerate(affected_flrgs):
                    if aset.get_key() in self.flrgs:
                        lower.append(self.flrgs[aset.get_key()].get_lower(tdisp) *
                                     affected_flrgs_memberships[ct])
                        upper.append(self.flrgs[aset.get_key()].get_upper(tdisp) *
                                     affected_flrgs_memberships[ct])
                    else:
                        lower.append(aset.LHS[-1].get_lower(tdisp) *
                                   affected_flrgs_memberships[ct])
                        upper.append(aset.LHS[-1].get_upper(tdisp) *
                                   affected_flrgs_memberships[ct])

            ret.append([sum(lower), sum(upper)])


        ret = self.apply_inverse_transformations(ret, params=[data[self.order - 1:]])

        return ret
