import numpy as np
import math
import random as rnd
import functools, operator
from pyFTS.common import FuzzySet, Membership
from pyFTS.partitioners import partitioner


class GridPartitioner(partitioner.Partitioner):
    """Even Length Grid Partitioner"""
    def __init__(self, data, npart, func = Membership.trimf, transformation=None, indexer=None):
        super(GridPartitioner, self).__init__("Grid", data, npart, func=func, transformation=transformation, indexer=indexer)

    def build(self, data):
        sets = []

        dlen = self.max - self.min
        partlen = dlen / self.partitions

        count = 0
        for c in np.linspace(self.min, self.max, self.partitions):
            if self.membership_function == Membership.trimf:
                sets.append(
                    FuzzySet.FuzzySet(self.prefix + str(count), Membership.trimf, [c - partlen, c, c + partlen],c))
            elif self.membership_function == Membership.gaussmf:
                sets.append(
                    FuzzySet.FuzzySet(self.prefix + str(count), Membership.gaussmf, [c, partlen / 3], c))
            elif self.membership_function == Membership.trapmf:
                q = partlen / 2
                sets.append(
                    FuzzySet.FuzzySet(self.prefix + str(count), Membership.trapmf, [c - partlen, c - q, c + q, c + partlen], c))
            count += 1

        self.min = self.min - partlen

        return sets
