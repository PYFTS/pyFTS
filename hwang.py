import numpy as np
from pyFTS.common import FuzzySet,FLR,Transformations
from pyFTS import fts


class HighOrderFTS(fts.FTS):
    def __init__(self, order, name):
        super(HighOrderFTS, self).__init__(order, name)

    def forecast(self, data, t):
        cn = np.array([0.0 for k in range(len(self.sets))])
        ow = np.array([[0.0 for k in range(len(self.sets))] for z in range(self.order - 1)])
        rn = np.array([[0.0 for k in range(len(self.sets))] for z in range(self.order - 1)])
        ft = np.array([0.0 for k in range(len(self.sets))])

        for s in range(len(self.sets)):
            cn[s] = self.sets[s].membership(data[t])
            for w in range(self.order - 1):
                ow[w, s] = self.sets[s].membership(data[t - w])
                rn[w, s] = ow[w, s] * cn[s]
                ft[s] = max(ft[s], rn[w, s])
        mft = max(ft)
        out = 0.0
        count = 0.0
        for s in range(len(self.sets)):
            if ft[s] == mft:
                out = out + self.sets[s].centroid
                count = count + 1.0
        return out / count

    def train(self, data, sets):
        self.sets = sets

    def predict(self, data, t):
        return self.forecast(data, t)

    def predictDiff(self, data, t):
        return data[t] + self.forecast(Transformations.differential(data), t)
