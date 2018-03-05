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
    def __init__(self, name, **kwargs):
        super(HighOrderFTS, self).__init__(1, name="HOFTS" + name, **kwargs)
        self.name = "High Order FTS"
        self.shortname = "HOFTS" + name
        self.detail = "Chen"
        self.order = kwargs.get('order',1)
        self.setsDict = {}
        self.is_high_order = True

    def generate_lhs_flrg(self, sample):
        lags = {}

        flrgs = []

        for o in np.arange(0, self.order):
            lhs = [key for key in self.partitioner.ordered_sets if self.sets[key].membership(sample[o]) > 0.0]
            lags[o] = lhs

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
        for k in np.arange(self.order, l):
            if self.dump: print("FLR: " + str(k))

            sample = data[k - self.order: k]

            rhs = [key for key in self.partitioner.ordered_sets if self.sets[key].membership(data[k]) > 0.0]

            flrgs = self.generate_lhs_flrg(sample)

            for flrg in flrgs:
                if flrg.get_key() not in self.flrgs:
                    self.flrgs[flrg.get_key()] = flrg;

                for st in rhs:
                    self.flrgs[flrg.get_key()].append_rhs(st)


    def train(self, data, **kwargs):

        data = self.apply_transformations(data, updateUoD=True)

        self.order = kwargs.get('order',2)

        if kwargs.get('sets', None) is not None:
            self.sets = kwargs.get('sets', None)

        self.generate_flrg(data)

    def forecast(self, data, **kwargs):

        ret = []

        l = len(data)

        if l <= self.order:
            return data

        ndata = self.apply_transformations(data)

        for k in np.arange(self.order, l+1):
            flrgs = self.generate_lhs_flrg(ndata[k - self.order: k])

            for flrg in flrgs:
                tmp = []
                if flrg.get_key() not in self.flrgs:
                    tmp.append(self.sets[flrg.LHS[-1]].centroid)
                else:
                    flrg = self.flrgs[flrg.get_key()]
                    tmp.append(flrg.get_midpoint(self.sets))

            ret.append(np.nanmean(tmp))

        ret = self.apply_inverse_transformations(ret, params=[data[self.order - 1:]])

        return ret
