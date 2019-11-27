import numpy as np
from pyFTS.common import FuzzySet, FLR, fts
from pyFTS.models import hofts
from pyFTS.models.nonstationary import common, flrg, nsfts
from itertools import product


class HighOrderNonStationaryFLRG(flrg.NonStationaryFLRG):
    """First Order NonStationary Fuzzy Logical Relationship Group"""
    def __init__(self, order, **kwargs):
        super(HighOrderNonStationaryFLRG, self).__init__(order, **kwargs)

        self.LHS = []
        self.RHS = {}
        self.count = 0.0
        self.strlhs = ""
        self.w = None

    def append_rhs(self, fset, **kwargs):
        count = kwargs.get('count',1.0)
        if fset not in self.RHS:
            self.RHS[fset] = count
        else:
            self.RHS[fset] += count
        self.count += count

    def append_lhs(self, c):
        self.LHS.append(c)

    def weights(self):
        if self.w is None:
            self.w = np.array([self.RHS[c] / self.count for c in self.RHS.keys()])
        return self.w

    def get_midpoint(self, sets, perturb):
        mp = np.array([sets[c].get_midpoint(perturb) for c in self.RHS.keys()])
        midpoint = mp.dot(self.weights())
        return midpoint

    def get_lower(self, sets, perturb):
        lw = np.array([sets[s].get_lower(perturb) for s in self.RHS.keys()])
        lower = lw.dot(self.weights())
        return lower

    def get_upper(self, sets, perturb):
        up = np.array([sets[s].get_upper(perturb) for s in self.RHS.keys()])
        upper = up.dot(self.weights())
        return upper

    def __str__(self):
        _str = ""
        for k in self.RHS.keys():
            _str += ", " if len(_str) > 0 else ""
            _str += k + " (" + str(round(self.RHS[k] / self.count, 3)) + ")"

        return self.get_key() + " -> " + _str

    def __len__(self):
        return len(self.RHS)


class HighOrderNonStationaryFTS(nsfts.NonStationaryFTS):
    """NonStationaryFTS Fuzzy Time Series"""
    def __init__(self, **kwargs):
        super(HighOrderNonStationaryFTS, self).__init__(**kwargs)
        self.name = "High Order Non Stationary FTS"
        self.shortname = "HONSFTS"
        self.detail = ""
        self.flrgs = {}
        self.is_high_order = True
        self.order = kwargs.get("order",2)
        self.configure_lags(**kwargs)

    def configure_lags(self, **kwargs):
        if "order" in kwargs:
            self.order = kwargs.get("order", self.min_order)

        if "lags" in kwargs:
            self.lags = kwargs.get("lags", None)

        if self.lags is not None:
            self.max_lag = max(self.lags)
        else:
            self.max_lag = self.order
            self.lags = np.arange(1, self.order + 1)

    def train(self, data, **kwargs):

        self.generate_flrg(data)

        if self.method == 'conditional':
            self.forecasts = self.forecast(data, no_update=True)
            self.residuals = np.array(data[self.order:]) - np.array(self.forecasts)

            self.variance_residual = np.var(self.residuals)  # np.max(self.residuals
            self.mean_residual = np.mean(self.residuals)

            self.residuals = self.residuals[-self.memory_window:].tolist()
            self.forecasts = self.forecasts[-self.memory_window:]
            self.inputs = np.array(data[-self.memory_window:]).tolist()

    def generate_flrg(self, data, **kwargs):
        l = len(data)
        for k in np.arange(self.order, l):
            if self.dump: print("FLR: " + str(k))

            sample = data[k - self.order: k]

            rhs = [key for key in self.partitioner.ordered_sets
                   if self.partitioner.sets[key].membership(data[k], [0,1]) > 0.0]

            if len(rhs) == 0:
                rhs = [common.check_bounds(data[k], self.partitioner, [0,1]).name]

            lags = []

            for o in np.arange(0, self.order):
                tdisp = [0,1]
                lhs = [key for key in self.partitioner.ordered_sets
                   if self.partitioner.sets[key].membership(sample[o], tdisp) > 0.0]

                if len(lhs) == 0:
                    lhs = [common.check_bounds(sample[o], self.partitioner, tdisp).name]

                lags.append(lhs)

            # Trace the possible paths
            for path in product(*lags):
                flrg = HighOrderNonStationaryFLRG(self.order)

                for c, e in enumerate(path, start=0):
                    flrg.append_lhs(e)

                if flrg.get_key() not in self.flrgs:
                    self.flrgs[flrg.get_key()] = flrg;

                for st in rhs:
                    self.flrgs[flrg.get_key()].append_rhs(st)

    def _affected_flrgs(self, sample, perturb):

        affected_flrgs = []
        affected_flrgs_memberships = []

        lags = []

        for ct, dat in enumerate(sample):
            affected_sets = [[ct, self.partitioner.sets[key].membership(dat, perturb[ct])]
                             for ct, key in enumerate(self.partitioner.ordered_sets)
                             if self.partitioner.sets[key].membership(dat, perturb[ct]) > 0.0]

            if len(affected_sets) == 0:

                if dat < self.partitioner.lower_set().get_lower(perturb[0]):
                    affected_sets.append([0, 1])
                elif dat > self.partitioner.upper_set().get_upper(perturb[-1]):
                    affected_sets.append([self.partitioner.partitions - 1, 1])

            lags.append([a[0] for a in affected_sets])

        # Build the tree with all possible paths

        # Trace the possible paths
        for path in product(*lags):

            flrg = HighOrderNonStationaryFLRG(self.order)

            for kk in path:
                flrg.append_lhs(self.partitioner.ordered_sets[kk])

            affected_flrgs.append(flrg)
            mv = []
            for ct, dat in enumerate(sample):
                fset = self.partitioner.sets[flrg.LHS[ct]]
                ix = self.partitioner.ordered_sets.index(flrg.LHS[ct])
                tmp = fset.membership(dat, perturb[ix])

                mv.append(tmp)

            affected_flrgs_memberships.append(np.prod(mv))

        return [affected_flrgs, affected_flrgs_memberships]

    def forecast(self, ndata, **kwargs):

        explain = kwargs.get('explain', False)

        fuzzyfied = kwargs.get('fuzzyfied', False)

        time_displacement = kwargs.get("time_displacement", 0)

        window_size = kwargs.get("window_size", 1)

        no_update = kwargs.get("no_update", False)

        mode = kwargs.get('mode', 'mean')

        ret = []

        l = len(ndata) if not explain else self.max_lag + 1

        if l < self.max_lag:
            return ndata
        elif l == self.max_lag:
            l += 1

        for k in np.arange(self.max_lag, l):

            sample = ndata[k - self.max_lag: k]

            if self.method == 'unconditional':
                perturb = common.window_index(k + time_displacement, window_size)
            elif self.method == 'conditional':
                if no_update:
                    perturb = [[0, 1] for k in np.arange(self.partitioner.partitions)]
                else:
                    perturb = self.conditional_perturbation_factors(sample[0])


            affected_flrgs, affected_flrgs_memberships = self._affected_flrgs(sample, perturb)

            tmp = []

            perturb2 = perturb[0]
            if len(affected_flrgs) == 0:
                tmp.append(common.check_bounds(sample[-1], self.partitioner.sets, perturb2))
            elif len(affected_flrgs) == 1:
                flrg = affected_flrgs[0]
                if flrg.get_key() in self.flrgs:
                    tmp.append(self.flrgs[flrg.get_key()].get_midpoint(self.partitioner.sets, perturb2))
                else:
                    fset = self.partitioner.sets[flrg.LHS[-1]]
                    ix = self.partitioner.ordered_sets.index(flrg.LHS[-1])
                    tmp.append(fset.get_midpoint(perturb[ix]))
            else:
                for ct, aset in enumerate(affected_flrgs):
                    if aset.get_key() in self.flrgs:

                        tmp.append(self.flrgs[aset.get_key()].get_midpoint(self.partitioner.sets, perturb2) *
                                   affected_flrgs_memberships[ct])
                    else:
                        fset = self.partitioner.sets[aset.LHS[-1]]
                        ix = self.partitioner.ordered_sets.index(aset.LHS[-1])
                        tmp.append(fset.get_midpoint(perturb[ix])*affected_flrgs_memberships[ct])
            pto = sum(tmp)

            ret.append(pto)

        return ret
