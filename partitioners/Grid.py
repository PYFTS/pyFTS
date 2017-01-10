import numpy as np
import math
import random as rnd
import functools, operator
from pyFTS.common import FuzzySet, Membership


# print(common.__dict__)

def GridPartitionerTrimf(data, npart, names=None, prefix="A"):
    sets = []
    dmax = max(data)
    dmax += dmax * 0.1
    print(dmax)
    dmin = min(data)
    dmin -= dmin * 0.1
    print(dmin)
    dlen = dmax - dmin
    partlen = math.ceil(dlen / npart)
    #p2 = partlen / 2
    #partition = dmin #+ partlen
    count = 0
    for c in np.arange(dmin, dmax, partlen):
        sets.append(
            FuzzySet.FuzzySet(prefix + str(count), Membership.trimf, [c - partlen, c, c + partlen],c))
        count += 1
        #partition += partlen

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
