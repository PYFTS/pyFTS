import numpy as np
import math
import random as rnd
import functools, operator
from pyFTS.common import FuzzySet, Membership, Transformations


# K. H. Huarng, “Effective lengths of intervals to improve forecasting in fuzzy time series,”
# Fuzzy Sets Syst., vol. 123, no. 3, pp. 387–394, Nov. 2001.

def GridPartitionerTrimf(data, prefix="A"):
    data2 = Transformations.differential(data)
    davg = np.abs( np.mean(data2) / 2 )

    if davg <= 1.0:
        base = 0.1
    elif 1 < davg <= 10:
        base = 1.0
    elif 10 < davg <= 100:
        base = 10
    else:
        base = 100

    sets = []
    dmax = max(data)
    dmax += dmax * 0.10
    dmin = min(data)
    dmin -= dmin * 0.10
    dlen = dmax - dmin
    npart = math.ceil(dlen / base)
    partition = math.ceil(dmin)
    for c in range(npart):
        sets.append(
            FuzzySet.FuzzySet(prefix + str(c), Membership.trimf, [partition - base, partition, partition + base], partition))
        partition += base

    return sets
