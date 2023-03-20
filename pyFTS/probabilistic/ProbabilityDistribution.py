import numpy as np
import matplotlib.pyplot as plt
from pyFTS.common import SortedCollection 
from pyFTS.probabilistic import kde


def from_point(x,**kwargs):
    """
    Create a probability distribution from a scalar value

    :param x: scalar value
    :param kwargs: common parameters of the distribution
    :return: the ProbabilityDistribution object
    """
    tmp = ProbabilityDistribution(**kwargs)
    tmp.set(x, 1.0)
    return tmp


class ProbabilityDistribution(object):
    """
    Represents a discrete or continous probability distribution
    If type is histogram, the PDF is discrete
    If type is KDE the PDF is continuous
    """
    def __init__(self, type = "KDE", **kwargs):
        self.uod = kwargs.get("uod", None)
        """Universe of discourse"""

        self.data = []

        data = kwargs.get("data", None)

        self.type = type
        """
        If type is histogram, the PDF is discrete
        If type is KDE the PDF is continuous
        """

        self.bins = kwargs.get("bins", None)
        """Number of bins on a discrete PDF"""
        self.labels = kwargs.get("bins_labels", None)
        """Bins labels on a discrete PDF"""

        if self.type == "KDE":
            self.kde = kde.KernelSmoothing(h=kwargs.get("h", 0.5), kernel=kwargs.get("kernel", "epanechnikov"))

        if data is not None and self.uod is None:
            _min = np.nanmin(data)
            _min = _min * .7 if _min > 0 else _min * 1.3
            _max = np.nanmax(data)
            _max = _max * 1.3 if _max > 0 else _max * .7
            self.uod = [_min, _max]

        self.nbins = kwargs.get("num_bins", 100)

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

        if data is not None:
            self.append(data)

        self.name = kwargs.get("name", "")

    def set(self, value, density):
        """
        Assert a probability 'density' for a certain value 'value', such that P(value) = density

        :param value: A value in the universe of discourse from the distribution
        :param density: The probability density to assign to the value
        """
        k = self.bin_index.find_ge(value)
        self.distribution[k] = density

    def append(self, values):
        """
        Increment the frequency count for the values

        :param values: A list of values to account the frequency
        """
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
        """
        Increment the frequency count for all values inside an interval

        :param intervals: A list of intervals do increment the frequency
        """
        if self.type == "histogram":
            for interval in intervals:
                for k in self.bin_index.inside(interval[0], interval[1]):
                    self.distribution[k] += 1
                    self.count += 1

    def density(self, values):
        """
        Return the probability densities for the input values

        :param values: List of values to return the densities
        :return: List of probability densities for the input values
        """
        ret = []
        scalar = False

        if not isinstance(values, list):
            values = [values]
            scalar = True

        for k in values:
            if self.type == "histogram":
                v = self.bin_index.find_ge(k)
                ret.append(self.distribution[v] / (self.count + 1e-5))
            elif self.type == "KDE":
                v = self.kde.probability(k, data=self.data)
                ret.append(v)
            else:
                v = self.bin_index.find_ge(k)
                ret.append(self.distribution[v])

        if scalar:
            return ret[0]

        return ret

    def differential_offset(self, value):
        """
        Auxiliary function for probability distributions of differentiated data

        :param value:
        :return:
        """
        nbins = []
        dist = {}

        for k in self.bins:
            nk = k+value
            nbins.append(nk)
            dist[nk] = self.distribution[k]

        self.bins = nbins
        self.distribution = dist
        self.labels = [str(k) for k in self.bins]

        self.bin_index = SortedCollection.SortedCollection(iterable=sorted(self.bins))
        self.quantile_index = None
        self.cdf = None
        self.qtl = None

    def expected_value(self):
        """
        Return the expected value of the distribution, as E[X] = ∑ x * P(x)

        :return: The expected value of the distribution
        """
        return np.nansum([v * self.distribution[v] for v in self.bins])

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

    def cumulative(self, values):
        """
        Return the cumulative probability densities for the input values,
        such that F(x) = P(X <= x)

        :param values: A list of input values
        :return: The cumulative probability densities for the input values
        """
        if self.cdf is None:
            self.build_cdf_qtl()

        if isinstance(values, list):
            ret = []
            for val in values:
                try:
                    k = self.bin_index.find_ge(val)
                    #ret.append(self.cdf[k])
                    ret.append(self.cdf[val])
                except:
                    ret.append(np.nan)
        else:
            try:
                k = self.bin_index.find_ge(values)
                return self.cdf[k]
            except:
                return np.nan

    def quantile(self, values):
        """
        Return the Universe of Discourse values in relation to the quantile input values,
        such that Q(tau) = min( {x | F(x) >= tau })

        :param values: input values
        :return: The list of the quantile values for the input values
        """
        if self.qtl is None:
            self.build_cdf_qtl()

        if isinstance(values, list):
            ret = []
            for val in values:
                try:
                    k = self.quantile_index.find_ge(val)
                    ret.append(self.qtl[str(k)][0])
                except:
                    ret.append(np.nan)
        else:
            try:
                k = self.quantile_index.find_ge(values)
                ret = self.qtl[str(k)]
            except:
                return np.nan

        return ret

    def entropy(self):
        """
        Return the entropy of the probability distribution, H(P) = E[ -ln P(X) ] = - ∑ P(x) log ( P(x) )

        :return:the entropy of the probability distribution
        """
        h = -np.nansum([self.distribution[k] * np.log(self.distribution[k]) if self.distribution[k] > 0 else 0
                  for k in self.bins])
        return h

    def crossentropy(self,q):
        """
        Cross entropy between the actual probability distribution and the informed one, 
        H(P,Q) = - ∑ P(x) log ( Q(x) )

        :param q: a probabilistic.ProbabilityDistribution object
        :return: Cross entropy between this probability distribution and the given distribution
        """
        h = -np.nansum([self.distribution[k] * np.log(q.distribution[k]) if self.distribution[k] > 0 else 0
                  for k in self.bins])
        return h

    def kullbackleiblerdivergence(self,q):
        """
        Kullback-Leibler divergence between the actual probability distribution and the informed one.
        DKL(P || Q) = - ∑ P(x) log( P(X) / Q(x) )

        :param q:  a probabilistic.ProbabilityDistribution object
        :return: Kullback-Leibler divergence
        """
        h = np.nansum([self.distribution[k] * np.log(self.distribution[k]/q.distribution[k]) if self.distribution[k] > 0 else 0
                  for k in self.bins])
        return h

    def empiricalloglikelihood(self):
        """
        Empirical Log Likelihood of the probability distribution, L(P) = ∑ log( P(x) )

        :return:
        """
        _s = 0
        for k in self.bins:
            if self.distribution[k] > 0:
                _s += np.log(self.distribution[k])
        return _s

    def pseudologlikelihood(self, data):
        """
        Pseudo log likelihood of the probability distribution with respect to data

        :param data:
        :return:
        """

        densities = self.density(data)

        _s = 0
        for k in densities:
            if k > 0:
                _s += np.log(k)
        return _s

    def averageloglikelihood(self, data):
        """
        Average log likelihood of the probability distribution with respect to data

        :param data:
        :return:
        """

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
        for k in sorted(self.bins):
            ret += str(round(k,2)) + ':\t'
            if self.type == "histogram":
                ret +=  str(round(self.distribution[k]  / self.count,3))
            elif self.type == "KDE":
                ret +=  str(round(self.density(k),3))
            else:
                ret += str(round(self.distribution[k], 6))
            ret += '\n'
        return ret
