"""
High Order FTS

Severiano, S. A. Jr; Silva, P. C. L.; Sadaei, H. J.; Guimarães, F. G. Very Short-term Solar Forecasting
using Fuzzy Time Series. 2017 IEEE International Conference on Fuzzy Systems. DOI10.1109/FUZZ-IEEE.2017.8015732
"""

import numpy as np
from pyFTS.common import FuzzySet, FLR, fts, flrg
from itertools import product


class HighOrderFLRG(flrg.FLRG):
    """Conventional High Order Fuzzy Logical Relationship Group"""
    def __init__(self, order, **kwargs):
        super(HighOrderFLRG, self).__init__(order, **kwargs)
        self.LHS = []
        self.RHS = {}
        self.strlhs = ""

    def append_rhs(self, c, **kwargs):
        if c not in self.RHS:
            self.RHS[c] = c

    def append_lhs(self, c):
        self.LHS.append(c)

    def __str__(self):
        tmp = ""
        for c in sorted(self.RHS):
            if len(tmp) > 0:
                tmp = tmp + ","
            tmp = tmp + c
        return self.get_key() + " -> " + tmp


    def __len__(self):
        return len(self.RHS)


class WeightedHighOrderFLRG(flrg.FLRG):
    """Weighted High Order Fuzzy Logical Relationship Group"""

    def __init__(self, order, **kwargs):
        super(WeightedHighOrderFLRG, self).__init__(order, **kwargs)
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

    def get_midpoint(self, sets):
        if self.midpoint is None:
            mp = np.array([sets[c].centroid for c in self.RHS.keys()])
            self.midpoint = mp.dot(self.weights())

        return self.midpoint

    def get_lower(self, sets):
        if self.lower is None:
            lw = np.array([sets[s].lower for s in self.RHS.keys()])
            self.lower = lw.dot(self.weights())
        return self.lower

    def get_upper(self, sets):
        if self.upper is None:
            up = np.array([sets[s].upper for s in self.RHS.keys()])
            self.upper = up.dot(self.weights())
        return self.upper

    def __str__(self):
        _str = ""
        for k in self.RHS.keys():
            _str += ", " if len(_str) > 0 else ""
            _str += k + " (" + str(round(self.RHS[k] / self.count, 3)) + ")"

        return self.get_key() + " -> " + _str

    def __len__(self):
        return len(self.RHS)


class HighOrderFTS(fts.FTS):
    """Conventional High Order Fuzzy Time Series"""
    def __init__(self, **kwargs):
        super(HighOrderFTS, self).__init__(**kwargs)
        self.name = "High Order FTS"
        self.shortname = "HOFTS"
        self.detail = "Severiano, Silva, Sadaei and Guimarães"
        self.is_high_order = True
        self.min_order = 1
        self.order= kwargs.get("order", self.min_order)
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
            self.lags = np.arange(1, self.order+1)

    def generate_lhs_flrg(self, sample, explain=False):

        nsample = [self.partitioner.fuzzyfy(k, mode="sets", alpha_cut=self.alpha_cut)
                   for k in sample]

        if explain:
            self.append_log("Fuzzyfication","{} -> {}".format(sample, nsample))

        return self.generate_lhs_flrg_fuzzyfied(nsample, explain)

    def generate_lhs_flrg_fuzzyfied(self, sample, explain=False):
        lags = []
        flrgs = []

        for ct, o in enumerate(self.lags):
            lhs = sample[o - 1]
            lags.append(lhs)

            if explain:
                self.append_log("Ordering Lags", "Lag {} Value {}".format(o, lhs))

        # Trace the possible paths
        for path in product(*lags):
            flrg = HighOrderFLRG(self.order)

            for lhs in path:
                flrg.append_lhs(lhs)

            flrgs.append(flrg)

        return flrgs

    def generate_flrg(self, data):
        l = len(data)
        for k in np.arange(self.max_lag, l):

            if self.dump: print("FLR: " + str(k))

            sample = data[k - self.max_lag: k]

            rhs = self.partitioner.fuzzyfy(data[k], mode="sets", alpha_cut=self.alpha_cut)

            flrgs = self.generate_lhs_flrg(sample)

            for flrg in flrgs:
                if flrg.get_key() not in self.flrgs:
                    self.flrgs[flrg.get_key()] = flrg;

                for st in rhs:
                    self.flrgs[flrg.get_key()].append_rhs(st)


    def generate_flrg_fuzzyfied(self, data):
        l = len(data)
        for k in np.arange(self.max_lag, l):
            if self.dump: print("FLR: " + str(k))

            sample = data[k - self.max_lag: k]

            rhs = data[k]

            flrgs = self.generate_lhs_flrg_fuzzyfied(sample)

            for flrg in flrgs:

                if flrg.get_key() not in self.flrgs:
                    self.flrgs[flrg.get_key()] = flrg

                for st in rhs:
                    self.flrgs[flrg.get_key()].append_rhs(st)

    def train(self, data, **kwargs):
        self.configure_lags(**kwargs)
        if not kwargs.get('fuzzyfied',False):
            self.generate_flrg(data)
        else:
            self.generate_flrg_fuzzyfied(data)

    def forecast(self, ndata, **kwargs):

        explain = kwargs.get('explain', False)

        fuzzyfied = kwargs.get('fuzzyfied', False)

        mode = kwargs.get('mode', 'mean')

        ret = []

        l = len(ndata) if not explain else self.max_lag + 1

        if l < self.max_lag:
            return ndata
        elif l == self.max_lag:
            l += 1

        for k in np.arange(self.max_lag, l):

            sample = ndata[k - self.max_lag: k]

            if not fuzzyfied:
                flrgs = self.generate_lhs_flrg(sample, explain)
            else:
                flrgs = self.generate_lhs_flrg_fuzzyfied(sample, explain)

            midpoints = []
            memberships = []
            for flrg in flrgs:

                if flrg.get_key() not in self.flrgs:
                    if len(flrg.LHS) > 0:
                        mp = self.partitioner.sets[flrg.LHS[-1]].centroid
                        mv = self.partitioner.sets[flrg.LHS[-1]].membership(sample[-1]) if not fuzzyfied else None
                        midpoints.append(mp)
                        memberships.append(mv)

                        if explain:
                            self.append_log("Rule Matching", "{} -> {} (Naïve) Midpoint: {}".format(str(flrg.LHS), flrg.LHS[-1],
                                                                                            mp))
                else:
                    flrg = self.flrgs[flrg.get_key()]
                    mp = flrg.get_midpoint(self.partitioner.sets)
                    mv = flrg.get_membership(sample, self.partitioner.sets) if not fuzzyfied else None
                    midpoints.append(mp)
                    memberships.append(mv)

                    if explain:
                        self.append_log("Rule Matching", "{}, Midpoint: {} Membership: {}".format(flrg.get_key(), mp, mv))

            if mode == "mean" or fuzzyfied:
                final = np.nanmean(midpoints)
                if explain: self.append_log("Deffuzyfication", "By Mean: {}".format(final))
            else:
                final = np.dot(midpoints, memberships)/np.nansum(memberships)
                if explain: self.append_log("Deffuzyfication", "By Memberships: {}".format(final))

            ret.append(final)

        return ret


class WeightedHighOrderFTS(HighOrderFTS):
    """Weighted High Order Fuzzy Time Series"""
    def __init__(self, **kwargs):
        super(WeightedHighOrderFTS, self).__init__(**kwargs)
        self.name = "Weighted High Order FTS"
        self.shortname = "WHOFTS"

    def generate_lhs_flrg_fuzzyfied(self, sample, explain=False):
        lags = []
        flrgs = []

        for ct, o in enumerate(self.lags):
            lags.append(sample[o-1])

            if explain:
                print("\t (Lag {}) {} \n".format(o, sample[o-1]))

        # Trace the possible paths
        for path in product(*lags):
            flrg = WeightedHighOrderFLRG(self.order)

            for lhs in path:
                flrg.append_lhs(lhs)

            flrgs.append(flrg)

        return flrgs
