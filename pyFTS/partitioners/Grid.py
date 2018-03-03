import numpy as np
import math
import random as rnd
import functools, operator
from pyFTS.common import FuzzySet, Membership
from pyFTS.partitioners import partitioner


class GridPartitioner(partitioner.Partitioner):
    """Even Length Grid Partitioner"""

    def __init__(self, **kwargs):
        """
        Even Length Grid Partitioner
        :param data: Training data of which the universe of discourse will be extracted. The universe of discourse is the open interval between the minimum and maximum values of the training data.
        :param npart: The number of universe of discourse partitions, i.e., the number of fuzzy sets that will be created
        :param func: Fuzzy membership function (pyFTS.common.Membership)
        :param transformation: data transformation to be applied on data
        :param indexer:
        """
        super(GridPartitioner, self).__init__(name="Grid", **kwargs)

    def build(self, data):
        sets = {}

        kwargs = {'type': self.type, 'variable': self.variable}

        dlen = self.max - self.min
        partlen = dlen / self.partitions

        count = 0
        for c in np.arange(self.min, self.max, partlen):
            _name = self.get_name(count)
            if self.membership_function == Membership.trimf:
                sets[_name] = FuzzySet.FuzzySet(_name, Membership.trimf, [c - partlen, c, c + partlen],c,**kwargs)
            elif self.membership_function == Membership.gaussmf:
                sets[_name] = FuzzySet.FuzzySet(_name, Membership.gaussmf, [c, partlen / 3], c,**kwargs)
            elif self.membership_function == Membership.trapmf:
                q = partlen / 2
                sets[_name] = FuzzySet.FuzzySet(_name, Membership.trapmf, [c - partlen, c - q, c + q, c + partlen], c,**kwargs)
            count += 1

        self.min = self.min - partlen

        return sets
