import numpy as np
import math
import random as rnd
import functools,operator
from pyFTS.common import FuzzySet,Membership,Transformations

#print(common.__dict__)

def GridPartitionerTrimf(data,npart,names = None,prefix = "A"):
    data2 = Transformations.differential(data)
    davg = np.mean(data2)/2
    if davg <= 1.0:
        base = 0.1
    elif davg > 1 and davg <= 10:
        base = 1.0
    elif davg > 10 and davg <= 100:
        base = 10
    else:
        base = 100

    sets = []
    dmax = max(data)
    dmax = dmax + dmax*0.10
    dmin = min(data)
    dmin = dmin - dmin*0.10
    dlen = dmax - dmin
    partlen = math.ceil(dlen / npart)
    partition = math.ceil(dmin)
    for c in range(npart):
        sets.append(FuzzySet(prefix+str(c),Membership.trimf,[round(partition-partlen,3), partition, partition+partlen], partition ) )
        partition = partition + partlen

    return sets

