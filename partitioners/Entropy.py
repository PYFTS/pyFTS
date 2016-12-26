import numpy as np
import math
import random as rnd
import functools, operator
from pyFTS.common import FuzzySet, Membership


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
    return - sum([pmf[0] * math.log(pmf[0]), pmf[1] * math.log(pmf[1])])


def informationGain(data, thres1, thres2):
    return entropy(data, thres1) - entropy(data, thres2)


def bestSplit(data, npart):
    if len(data) < 2:
        return None
    count = 2
    ndata = list(set(data))
    ndata.sort()
    threshold = 0
    while informationGain(data, ndata[count - 1], ndata[count]) <= 0:
        threshold = ndata[count]
        count += 1

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

        return [ threshold, bestSplit(p1, np1 ), bestSplit(p2, np2 )  ]

    else:
        return threshold

def EntropyPartitionerTrimf(data, npart, prefix="A"):
    dmax = max(data)
    dmax += dmax * 0.10
    dmin = min(data)
    dmin -= dmin * 0.10

    sets = [dmin, bestSplit(data, npart), dmax]

    sets.sort()
    for c in np.arange(1, len(sets) - 1):
        sets.append(FuzzySet.FuzzySet(prefix + str(c), Membership.trimf,
                                      [round(sets[c - 1], 3), round(sets[c], 3),
                                       round(sets[c + 1], 3)],round(sets[c], 3)))

    return sets
