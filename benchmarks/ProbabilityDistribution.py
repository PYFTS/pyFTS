import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyFTS.common import FuzzySet,SortedCollection


class ProbabilityDistribution(object):
    def __init__(self,name,nbins,uod,bins=None,labels=None, data=None):
        self.name = name
        self.nbins = nbins
        self.uod = uod
        if bins is None:
            #range = (uod[1] - uod[0])/nbins
            #self.bins = np.arange(uod[0],uod[1],range).tolist()
            self.bins = np.linspace(uod[0], uod[1], nbins).tolist()
            self.labels = [str(k) for k in self.bins]
        else:
            self.bins = bins
            self.labels = labels

        self.index = SortedCollection.SortedCollection(iterable=sorted(self.bins))
        self.distribution = {}
        self.count = 0
        for k in self.bins: self.distribution[k] = 0

        if data is not None: self.append(data)

    def append(self, values):
        for k in values:
            v = self.index.find_ge(k)
            self.distribution[v] += 1
            self.count += 1

    def density(self, values):
        ret = []
        for k in values:
            v = self.index.find_ge(k)
            ret.append(self.distribution[v] / self.count)
        return ret

    def entropy(self):
        h = -sum([self.distribution[k] * np.log(self.distribution[k]) if self.distribution[k] > 0 else 0
                  for k in self.bins])
        return h

    def crossentropy(self,q):
        h = -sum([self.distribution[k] * np.log(q.distribution[k]) if self.distribution[k] > 0 else 0
                  for k in self.bins])
        return h

    def kullbackleiblerdivergence(self,q):
        h = sum([self.distribution[k] * np.log(self.distribution[k]/q.distribution[k]) if self.distribution[k] > 0 else 0
                  for k in self.bins])
        return h

    def empiricalloglikelihood(self):
        _s = 0
        for k in self.bins:
            if self.distribution[k] > 0:
                _s += np.log(self.distribution[k])
        return _s

    def pseudologlikelihood(self, data):

        densities = self.density(data)

        _s = 0
        for k in densities:
            if k > 0:
                _s += np.log(k)
        return _s

    def averageloglikelihood(self, data):

        densities = self.density(data)

        _s = 0
        for k in densities:
            if k > 0:
                _s += np.log(k)
        return _s / len(data)

    def plot(self,axis=None,color="black",tam=[10, 6]):
        if axis is None:
            fig = plt.figure(figsize=tam)
            axis = fig.add_subplot(111)

        ys = [self.distribution[k]/self.count for k in self.bins]

        axis.plot(self.bins, ys,c=color, label=self.name)

        axis.set_xlabel('Universe of Discourse')
        axis.set_ylabel('Probability')


    def __str__(self):
        head = '|'
        body = '|'
        for k in sorted(self.distribution.keys()):
            head += str(round(k,2)) + '\t|'
            body += str(round(self.distribution[k]  / self.count,3)) + '\t|'
        return head + '\n' + body
