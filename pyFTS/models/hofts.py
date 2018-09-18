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

    def generate_lhs_flrg(self, sample):
        lags = {}

        flrgs = []

        for ct, o in enumerate(self.lags):
            lhs = FuzzySet.fuzzyfy(sample[o-1], partitioner=self.partitioner, mode="sets", alpha_cut=self.alpha_cut)
            lags[ct] = lhs

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
            if self.dump: print("FLR: " + str(k))

            sample = data[k - self.max_lag: k]

            rhs = FuzzySet.fuzzyfy(data[k], partitioner=self.partitioner, mode="sets", alpha_cut=self.alpha_cut)

            flrgs = self.generate_lhs_flrg(sample)

            for flrg in flrgs:
                if flrg.get_key() not in self.flrgs:
                    self.flrgs[flrg.get_key()] = flrg;

                for st in rhs:
                    self.flrgs[flrg.get_key()].append_rhs(st)

    def train(self, data, **kwargs):
        self.configure_lags(**kwargs)
        self.generate_flrg(data)

    def forecast(self, ndata, **kwargs):

        ret = []

        l = len(ndata)

        if l < self.max_lag:
            return ndata

        for k in np.arange(self.max_lag, l+1):
            flrgs = self.generate_lhs_flrg(ndata[k - self.max_lag: k])

            tmp = []
            for flrg in flrgs:

                if flrg.get_key() not in self.flrgs:
                    if len(flrg.LHS) > 0:
                        tmp.append(self.sets[flrg.LHS[-1]].centroid)
                else:
                    flrg = self.flrgs[flrg.get_key()]
                    tmp.append(flrg.get_midpoint(self.sets))

            ret.append(np.nanmean(tmp))

        return ret
