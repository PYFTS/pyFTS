"""
High Order Fuzzy Time Series by Hwang, Chen and Lee (1998)

Jeng-Ren Hwang, Shyi-Ming Chen, and Chia-Hoang Lee, “Handling forecasting problems using fuzzy time series,” 
Fuzzy Sets Syst., no. 100, pp. 217–228, 1998.
"""

import numpy as np
from pyFTS.common import FuzzySet, FLR, Transformations, fts


class HighOrderFTS(fts.FTS):
    def __init__(self, **kwargs):
        super(HighOrderFTS, self).__init__(**kwargs)
        self.is_high_order = True
        self.min_order = 2
        self.name = "Hwang High Order FTS"
        self.shortname = "Hwang"
        self.detail = "Hwang"

    def forecast(self, ndata, **kwargs):

        if self.sets == None:
            self.sets = self.partitioner.sets
            ordered_sets = self.partitioner.ordered_sets
        else:
            ordered_sets = FuzzySet.set_ordered(self.sets)

        l = len(self.sets)

        cn = np.array([0.0 for k in range(l)])
        ow = np.array([[0.0 for k in range(l)] for z in range(self.order - 1)])
        rn = np.array([[0.0 for k in range(l)] for z in range(self.order - 1)])
        ft = np.array([0.0 for k in range(l)])

        ret = []

        for t in np.arange(self.order-1, len(ndata)):

            for ix in range(l):
                s = ordered_sets[ix]
                cn[ix] = self.sets[s].membership( FuzzySet.grant_bounds(ndata[t], self.sets, ordered_sets))
                for w in range(self.order - 1):
                    ow[w, ix] = self.sets[s].membership(FuzzySet.grant_bounds(ndata[t - w], self.sets, ordered_sets))
                    rn[w, ix] = ow[w, ix] * cn[ix]
                    ft[ix] = max(ft[ix], rn[w, ix])
            mft = max(ft)
            out = 0.0
            count = 0.0
            for ix in range(l):
                s = ordered_sets[ix]
                if ft[ix] == mft:
                    out = out + self.sets[s].centroid
                    count += 1.0
            ret.append(out / count)

        return ret

    def train(self, data, **kwargs):
        pass