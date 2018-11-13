"""
High Order FTS

Severiano, S. A. Jr; Silva, P. C. L.; Sadaei, H. J.; Guimarães, F. G. Very Short-term Solar Forecasting
using Fuzzy Time Series. 2017 IEEE International Conference on Fuzzy Systems. DOI10.1109/FUZZ-IEEE.2017.8015732
"""

import numpy as np
from pyFTS.common import FuzzySet, FLR, fts, flrg, tree

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
        if fset not in self.RHS:
            self.RHS[fset] = 1.0
        else:
            self.RHS[fset] += 1.0
        self.count += 1.0

    def append_lhs(self, c):
        self.LHS.append(c)

    def weights(self):
        if self.w is None:
            self.w = np.array([self.RHS[c] / self.count for c in self.RHS.keys()])
        return self.w

    def get_midpoint(self, sets):
        mp = np.array([sets[c].centroid for c in self.RHS.keys()])
        return mp.dot(self.weights())

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
        self.order= kwargs.get("order", 2)
        self.lags = kwargs.get("lags", None)
        self.configure_lags(**kwargs)

    def configure_lags(self, **kwargs):
        if "order" in kwargs:
            self.order = kwargs.get("order", 2)

        if "lags" in kwargs:
            self.lags = kwargs.get("lags", None)

        if self.lags is not None:
            self.max_lag = max(self.lags)
        else:
            self.max_lag = self.order
            self.lags = np.arange(1, self.order+1)

    def generate_lhs_flrg(self, sample, explain=False):

        nsample = [FuzzySet.fuzzyfy(k, partitioner=self.partitioner, mode="sets", alpha_cut=self.alpha_cut)
                   for k in sample]

        return self.generate_lhs_flrg_fuzzyfied(nsample, explain)

    def generate_lhs_flrg_fuzzyfied(self, sample, explain=False):
        lags = {}

        flrgs = []

        for ct, o in enumerate(self.lags):
            lags[ct] = sample[o-1]

            if explain:
                print("\t (Lag {}) {} -> {} \n".format(o, sample[o-1], lhs))

        root = tree.FLRGTreeNode(None)

        tree.build_tree_without_order(root, lags, 0)

        # Trace the possible paths
        for p in root.paths():
            flrg = HighOrderFLRG(self.order)
            path = list(reversed(list(filter(None.__ne__, p))))

            for lhs in path:
                flrg.append_lhs(lhs)

            flrgs.append(flrg)

        return flrgs

    def generate_flrg(self, data):
        l = len(data)
        for k in np.arange(self.max_lag, l):
            lags = {}

            if self.dump: print("FLR: " + str(k))

            sample = data[k - self.max_lag: k]

            rhs = FuzzySet.fuzzyfy(data[k], partitioner=self.partitioner, mode="sets", alpha_cut=self.alpha_cut)

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

        ret = []

        l = len(ndata) if not explain else self.max_lag + 1

        if l < self.max_lag:
            return ndata

        for k in np.arange(self.max_lag, l+1):

            if explain:
                print("Fuzzyfication \n")

            if not kwargs.get('fuzzyfied', False):
                flrgs = self.generate_lhs_flrg(ndata[k - self.max_lag: k], explain)
            else:
                flrgs = self.generate_lhs_flrg_fuzzyfied(ndata[k - self.max_lag: k], explain)

            if explain:
                print("Rules:\n")

            tmp = []
            for flrg in flrgs:

                if flrg.get_key() not in self.flrgs:
                    if len(flrg.LHS) > 0:
                        mp = self.partitioner.sets[flrg.LHS[-1]].centroid
                        tmp.append(mp)

                        if explain:
                            print("\t {} -> {} (Naïve)\t Midpoint: {}\n".format(str(flrg.LHS), flrg.LHS[-1],
                                                                                            mp))
                else:
                    flrg = self.flrgs[flrg.get_key()]
                    mp = flrg.get_midpoint(self.partitioner.sets)
                    tmp.append(mp)

                    if explain:
                        print("\t {} \t Midpoint: {}\n".format(str(flrg), mp))

            final = np.nanmean(tmp)
            ret.append(final)

            if explain:
                print("Deffuzyfied value: {} \n".format(final))

        return ret


class WeightedHighOrderFTS(HighOrderFTS):
    """Weighted High Order Fuzzy Time Series"""
    def __init__(self, **kwargs):
        super(WeightedHighOrderFTS, self).__init__(**kwargs)
        self.name = "Weighted High Order FTS"
        self.shortname = "WHOFTS"

    def generate_lhs_flrg_fuzzyfied(self, sample, explain=False):
        lags = {}

        flrgs = []

        for ct, o in enumerate(self.lags):
            lags[ct] = sample[o-1]

            if explain:
                print("\t (Lag {}) {} -> {} \n".format(o, sample[o-1], lhs))

        root = tree.FLRGTreeNode(None)

        tree.build_tree_without_order(root, lags, 0)

        # Trace the possible paths
        for p in root.paths():
            flrg = WeightedHighOrderFLRG(self.order)
            path = list(reversed(list(filter(None.__ne__, p))))

            for lhs in path:
                flrg.append_lhs(lhs)

            flrgs.append(flrg)

        return flrgs
