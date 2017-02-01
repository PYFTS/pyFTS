import numpy as np
import math
import random as rnd
import functools, operator
from pyFTS.common import FuzzySet, Membership


# print(common.__dict__)

def GridPartitionerTrimf(data, npart, names=None, prefix="A"):
    sets = []
    _min = min(data)
    if _min < 0:
        dmin = _min * 1.1
    else:
        dmin = _min * 0.9

    _max = max(data)
    if _max > 0:
        dmax = _max * 1.1
    else:
        dmax = _max * 0.9

    dlen = dmax - dmin
    partlen = dlen / npart

    count = 0
    for c in np.arange(dmin, dmax, partlen):
        sets.append(
            FuzzySet.FuzzySet(prefix + str(count), Membership.trimf, [c - partlen, c, c + partlen],c))
        count += 1

    return sets


def GridPartitionerGaussmf(data, npart, names=None, prefix="A"):
    sets = []
    dmax = max(data)
    dmax += dmax * 0.10
    dmin = min(data)
    dmin -= dmin * 0.10
    dlen = dmax - dmin
    partlen = math.ceil(dlen / npart)
    partition = math.ceil(dmin)
    for c in range(npart):
        sets.append(
            FuzzySet.FuzzySet(prefix + str(c), Membership.gaussmf, [partition, partlen/3],
                     partition))
        partition += partlen

    return sets
