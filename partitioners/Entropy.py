import numpy as np
import math
import random as rnd
import functools, operator
from pyFTS.common import FuzzySet, Membership
from pyFTS.partitioners import partitioner

# C. H. Cheng, R. J. Chang, and C. A. Yeh, “Entropy-based and trapezoidal fuzzification-based fuzzy time series approach for forecasting IT project cost,”
# Technol. Forecast. Social Change, vol. 73, no. 5, pp. 524–542, Jun. 2006.


def splitBelow(data,threshold):
    return [k for k in data if k <= threshold]


def splitAbove(data,threshold):
    return [k for k in data if k > threshold]


def PMF(data, threshold):
    a = sum([1.0 for k in splitBelow(data,threshold)])
    b = sum([1.0 for k in splitAbove(data, threshold)])
    l = len(data)
    return [a / l, b / l]


def entropy(data, threshold):
    pmf = PMF(data, threshold)
    if pmf[0] == 0 or pmf[1] == 0:
        return 1
    else:
        return - sum([pmf[0] * math.log(pmf[0]), pmf[1] * math.log(pmf[1])])


def informationGain(data, thres1, thres2):
    return entropy(data, thres1) - entropy(data, thres2)


def bestSplit(data, npart):
    if len(data) < 2:
        return None
    count = 1
    ndata = list(set(data))
    ndata.sort()
    l = len(ndata)
    threshold = 0
    try:
        while count < l and informationGain(data, ndata[count - 1], ndata[count]) <= 0:
            threshold = ndata[count]
            count += 1
    except IndexError:
        print(threshold)
        print (ndata)
        print (count)

    rem = npart % 2

    if (npart - rem)/2 > 1:
        p1 = splitBelow(data,threshold)
        p2 = splitAbove(data,threshold)

        if len(p1) > len(p2):
            np1 = (npart - rem)/2 + rem
            np2 = (npart - rem)/2
        else:
            np1 = (npart - rem) / 2
            np2 = (npart - rem) / 2 + rem

        tmp = [threshold]

        for k in bestSplit(p1, np1 ): tmp.append(k)
        for k in bestSplit(p2, np2 ): tmp.append(k)

        return tmp

    else:
        return [threshold]


class EntropyPartitioner(partitioner.Partitioner):
    def __init__(self, data, npart, func = Membership.trimf, transformation=None):
        super(EntropyPartitioner, self).__init__("Entropy", data, npart, func=func, transformation=transformation)

    def build(self, data):
        sets = []

        partitions = bestSplit(data, self.partitions)
        partitions.append(self.min)
        partitions.append(self.max)
        partitions = list(set(partitions))
        partitions.sort()
        for c in np.arange(1, len(partitions) - 1):
            if self.membership_function == Membership.trimf:
                sets.append(FuzzySet.FuzzySet(self.prefix + str(c), Membership.trimf,
                                              [partitions[c - 1], partitions[c], partitions[c + 1]],partitions[c]))
            elif self.membership_function == Membership.trapmf:
                b1 = (partitions[c] - partitions[c - 1])/2
                b2 = (partitions[c + 1] - partitions[c]) / 2
                sets.append(FuzzySet.FuzzySet(self.prefix + str(c), Membership.trapmf,
                                              [partitions[c - 1], partitions[c] - b1,
                                               partitions[c] - b2, partitions[c + 1]],
                                              partitions[c]))

        return sets
