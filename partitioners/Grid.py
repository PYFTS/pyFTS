import numpy as np
import math
import random as rnd
import functools, operator
from pyFTS.common import FuzzySet, Membership
from pyFTS.partitioners import partitioner


class GridPartitioner(partitioner.Partitioner):
    def __init__(self, data, npart, func = Membership.trimf, transformation=None):
        super(GridPartitioner, self).__init__("Grid", data, npart, func=func, transformation=transformation)

    def build(self, data):
        sets = []

        dlen = self.max - self.min
        partlen = dlen / self.partitions

        count = 0
        for c in np.arange(self.min, self.max, partlen):
            if self.membership_function == Membership.trimf:
                sets.append(
                    FuzzySet.FuzzySet(self.prefix + str(count), Membership.trimf, [c - partlen, c, c + partlen],c))
            elif self.membership_function == Membership.gaussmf:
                sets.append(
                    FuzzySet.FuzzySet(self.prefix + str(count), Membership.gaussmf, [c, partlen / 3], c))
            count += 1


        return sets
