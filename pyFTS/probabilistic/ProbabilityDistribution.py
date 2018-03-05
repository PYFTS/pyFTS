import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyFTS.common import FuzzySet,SortedCollection,tree
from pyFTS.probabilistic import kde


class ProbabilityDistribution(object):
    """
    Represents a discrete or continous probability distribution
    If type is histogram, the PDF is discrete
    If type is KDE the PDF is continuous
    """
    def __init__(self,type = "KDE", **kwargs):
        self.uod = kwargs.get("uod", None)

        self.type = type
        if self.type == "KDE":
            self.kde = kde.KernelSmoothing(kwargs.get("h", 0.5), kwargs.get("kernel", "epanechnikov"))

        self.nbins = kwargs.get("num_bins", 100)

        self.bins = kwargs.get("bins", None)
        self.labels = kwargs.get("bins_labels", None)

        if self.bins is None:
            self.bins = np.linspace(int(self.uod[0]), int(self.uod[1]), int(self.nbins)).tolist()
            self.labels = [str(k) for k in self.bins]

        if self.uod is not None:
            self.resolution = (self.uod[1] - self.uod[0]) / self.nbins

        self.bin_index = SortedCollection.SortedCollection(iterable=sorted(self.bins))
        self.quantile_index = None
        self.distribution = {}
        self.cdf = None
        self.qtl = None
        self.count = 0
        for k in self.bins: self.distribution[k] = 0

        self.data = []

        data = kwargs.get("data",None)

        if data is not None:
            self.append(data)

        self.name = kwargs.get("name", "")

    def set(self, value, density):
        k = self.bin_index.find_ge(value)
        self.distribution[k] = density

    def append(self, values):
        if self.type == "histogram":
            for k in values:
                v = self.bin_index.find_ge(k)
                self.distribution[v] += 1
                self.count += 1
        else:
            self.data.extend(values)
            self.distribution = {}
            dens = self.density(self.bins)
            for v,d in enumerate(dens):
                self.distribution[self.bins[v]] = d

    def append_interval(self, intervals):
        if self.type == "histogram":
            for interval in intervals:
                for k in self.bin_index.inside(interval[0], interval[1]):
                    self.distribution[k] += 1
                    self.count += 1

    def density(self, values):
        ret = []
        scalar = False

        if not isinstance(values, list):
            values = [values]
            scalar = True

        for k in values:
            if self.type == "histogram":
                v = self.bin_index.find_ge(k)
                ret.append(self.distribution[v] / self.count)
            elif self.type == "KDE":
                v = self.kde.probability(k, self.data)
                ret.append(v)
            else:
                v = self.bin_index.find_ge(k)
                ret.append(self.distribution[v])

        if scalar:
            return ret[0]

        return ret

    def build_cdf_qtl(self):
        ret = 0.0
        self.cdf = {}
        self.qtl = {}
        for k in sorted(self.bins):
            ret += self.density(k)
            if k not in self.cdf:
                self.cdf[k] = ret

            if str(ret) not in self.qtl:
                self.qtl[str(ret)] = []

            self.qtl[str(ret)].append(k)

        _keys = [float(k) for k in sorted(self.qtl.keys())]

        self.quantile_index = SortedCollection.SortedCollection(iterable=_keys)

    def cummulative(self, values):
        if self.cdf is None:
            self.build_cdf_qtl()

        if isinstance(values, list):
            ret = []
            for val in values:
                k = self.bin_index.find_ge(val)
                ret.append(self.cdf[k])
        else:
            k = self.bin_index.find_ge(values)
            return self.cdf[values]

    def quantile(self, values):
        if self.qtl is None:
            self.build_cdf_qtl()

        if isinstance(values, list):
            ret = []
            for val in values:
                k = self.quantile_index.find_ge(val)
                ret.append(self.qtl[str(k)][0])
        else:
            k = self.quantile_index.find_ge(values)
            ret = self.qtl[str(k)[0]]

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

    def plot(self,axis=None,color="black",tam=[10, 6], title = None):

        if axis is None:
            fig = plt.figure(figsize=tam)
            axis = fig.add_subplot(111)

        if self.type == "histogram":
            ys = [self.distribution[k]/self.count for k in self.bins]
        else:
            ys = [self.distribution[k] for k in self.bins]
            yp = [0 for k in self.data]
            axis.plot(self.data, yp, c="red")

        if title is None:
            title = self.name
        axis.plot(self.bins, ys, c=color)
        axis.set_title(title)

        axis.set_xlabel('Universe of Discourse')
        axis.set_ylabel('Probability')

    def __str__(self):
        ret = ""
        for k in sorted(self.distribution.keys()):
            ret += str(round(k,2)) + ':\t'
            if self.type == "histogram":
                ret +=  str(round(self.distribution[k]  / self.count,3))
            else:
                ret += str(round(self.distribution[k], 6))
            ret += '\n'
        return ret
