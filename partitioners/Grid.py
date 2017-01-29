import numpy as np
import math
import random as rnd
import functools, operator
from pyFTS.common import FuzzySet, Membership


# print(common.__dict__)

def GridPartitionerTrimf(data, npart, names=None, prefix="A"):
    sets = []
    if min(data) < 0:
        dmin = min(data) * 1.1
    else:
        dmin = min(data) * 0.9

    if max(data) > 0:
        dmax = max(data) * 1.1
    else:
        dmax = max(data) * 0.9

    dlen = dmax - dmin
    partlen = math.ceil(dlen / npart)

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
