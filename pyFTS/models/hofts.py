"""
Simple High Order extension of Conventional FTS by Chen (1996)

[1] S.-M. Chen, “Forecasting enrollments based on fuzzy time series,” 
Fuzzy Sets Syst., vol. 81, no. 3, pp. 311–319, 1996.
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

    def append_rhs(self, c):
        if c.name not in self.RHS:
            self.RHS[c.name] = c

    def str_lhs(self):
        if len(self.strlhs) == 0:
            for c in self.LHS:
                if len(self.strlhs) > 0:
                    self.strlhs += ", "
                self.strlhs = self.strlhs + str(c.name)
        return self.strlhs

    def append_lhs(self, c):
        self.LHS.append(c)

    def __str__(self):
        tmp = ""
        for c in sorted(self.RHS):
            if len(tmp) > 0:
                tmp = tmp + ","
            tmp = tmp + c
        return self.str_lhs() + " -> " + tmp


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

    def build_tree(self, node, lags, level):
        if level >= self.order:
            return

        for s in lags[level]:
            node.appendChild(tree.FLRGTreeNode(s))

        for child in node.getChildren():
            self.build_tree(child, lags, level + 1)

    def build_tree_without_order(self, node, lags, level):

        if level not in lags:
            return

        for s in lags[level]:
            node.appendChild(tree.FLRGTreeNode(s))

        for child in node.getChildren():
            self.build_tree_without_order(child, lags, level + 1)

    def generateFLRG(self, flrs):
        flrgs = {}
        l = len(flrs)
        for k in np.arange(self.order + 1, l):
            flrg = HighOrderFLRG(self.order)

            for kk in np.arange(k - self.order, k):
                flrg.append_lhs(flrs[kk].LHS)

            if flrg.str_lhs() in flrgs:
                flrgs[flrg.str_lhs()].append_rhs(flrs[k].RHS)
            else:
                flrgs[flrg.str_lhs()] = flrg;
                flrgs[flrg.str_lhs()].append_rhs(flrs[k].RHS)
        return (flrgs)

    def generate_flrg(self, data):
        flrgs = {}
        l = len(data)
        for k in np.arange(self.order, l):
            if self.dump: print("FLR: " + str(k))

            sample = data[k - self.order: k]

            rhs = [set for set in self.sets if set.membership(data[k]) > 0.0]

            lags = {}

            for o in np.arange(0, self.order):
                lhs = [set for set in self.sets if set.membership(sample[o]) > 0.0]

                lags[o] = lhs

            root = tree.FLRGTreeNode(None)

            self.build_tree_without_order(root, lags, 0)

            # Trace the possible paths
            for p in root.paths():
                flrg = HighOrderFLRG(self.order)
                path = list(reversed(list(filter(None.__ne__, p))))

                for lhs in path:
                    flrg.append_lhs(lhs)

                if flrg.str_lhs() not in flrgs:
                    flrgs[flrg.str_lhs()] = flrg;

                for st in rhs:
                    flrgs[flrg.str_lhs()].append_rhs(st)

        return flrgs

    def train(self, data, **kwargs):

        data = self.apply_transformations(data, updateUoD=True)

        self.order = kwargs.get('order',2)

        if kwargs.get('sets', None) is not None:
            self.sets = kwargs.get('sets', None)
        for s in self.sets:    self.setsDict[s.name] = s
        self.flrgs = self.generate_flrg(data)

    def forecast(self, data, **kwargs):

        ret = []

        l = len(data)

        if l <= self.order:
            return data

        ndata = self.apply_transformations(data)

        for k in np.arange(self.order, l+1):
            tmpdata = FuzzySet.fuzzyfy_series_old(ndata[k - self.order: k], self.sets)
            tmpflrg = HighOrderFLRG(self.order)

            for s in tmpdata: tmpflrg.append_lhs(s)

            if tmpflrg.str_lhs() not in self.flrgs:
                ret.append(tmpdata[-1].centroid)
            else:
                flrg = self.flrgs[tmpflrg.str_lhs()]
                ret.append(flrg.get_midpoint())

        ret = self.apply_inverse_transformations(ret, params=[data[self.order - 1:]])

        return ret
