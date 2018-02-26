"""
High Order Fuzzy Time Series by Hwang, Chen and Lee (1998)

Jeng-Ren Hwang, Shyi-Ming Chen, and Chia-Hoang Lee, “Handling forecasting problems using fuzzy time series,” 
Fuzzy Sets Syst., no. 100, pp. 217–228, 1998.
"""

import numpy as np
from pyFTS.common import FuzzySet,FLR,Transformations
from pyFTS import fts


class HighOrderFTS(fts.FTS):
    def __init__(self, name, **kwargs):
        super(HighOrderFTS, self).__init__(1, name, **kwargs)
        self.is_high_order = True
        self.min_order = 2
        self.name = "Hwang High Order FTS"
        self.shortname = "Hwang" + name
        self.detail = "Hwang"

    def forecast(self, data, **kwargs):

        ndata = self.apply_transformations(data)

        cn = np.array([0.0 for k in range(len(self.sets))])
        ow = np.array([[0.0 for k in range(len(self.sets))] for z in range(self.order - 1)])
        rn = np.array([[0.0 for k in range(len(self.sets))] for z in range(self.order - 1)])
        ft = np.array([0.0 for k in range(len(self.sets))])

        ret = []

        for t in np.arange(self.order-1, len(ndata)):

            for s in range(len(self.sets)):
                cn[s] = self.sets[s].membership(ndata[t])
                for w in range(self.order - 1):
                    ow[w, s] = self.sets[s].membership(ndata[t - w])
                    rn[w, s] = ow[w, s] * cn[s]
                    ft[s] = max(ft[s], rn[w, s])
            mft = max(ft)
            out = 0.0
            count = 0.0
            for s in range(len(self.sets)):
                if ft[s] == mft:
                    out = out + self.sets[s].centroid
                    count += 1.0
            ret.append(out / count)

        ret = self.apply_inverse_transformations(ret, params=[data[self.order - 1:]])

        return ret

    def train(self, data, sets, order=1, parameters=None):
        self.sets = sets
        self.order = order