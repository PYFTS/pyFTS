import numpy as np
import math
import random as rnd
import functools, operator
from pyFTS.common import FuzzySet, Membership


# print(common.__dict__)

def GridPartitionerTrimf(data, npart, names=None, prefix="A"):
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
            FuzzySet.FuzzySet(prefix + str(c), Membership.trimf, [round(partition - partlen, 3), partition, partition + partlen],
                     partition))
        partition += partlen

    return sets
