import numpy as np
import math
import random as rnd
import functools, operator
from pyFTS.common import FuzzySet, Membership
from pyFTS.partitioners import partitioner


class GridPartitioner(partitioner.Partitioner):
    def __init__(self, data,npart,func = Membership.trimf):
        super(GridPartitioner, self).__init__("Grid",data,npart,func)

    def build(self, data):
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
        partlen = dlen / self.partitions

        count = 0
        for c in np.arange(dmin, dmax, partlen):
            if self.membership_function == Membership.trimf:
                sets.append(
                    FuzzySet.FuzzySet(self.prefix + str(count), Membership.trimf, [c - partlen, c, c + partlen],c))
            elif self.membership_function == Membership.gaussmf:
                sets.append(
                    FuzzySet.FuzzySet(self.prefix + str(count), Membership.gaussmf, [c, partlen / 3], c))
            count += 1


        return sets
