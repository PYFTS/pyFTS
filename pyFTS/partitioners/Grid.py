"""Even Length Grid Partitioner"""

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
        """
        super(GridPartitioner, self).__init__(name="Grid", **kwargs)

    def build(self, data):
        sets = {}

        kwargs = {'type': self.type, 'variable': self.variable}

        #dlen = self.max - self.min
        #partlen = dlen / self.partitions

        count = 0
        centers, partlen = np.linspace(self.min, self.max, self.partitions, retstep=True)
        for c in centers:
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


class PreFixedGridPartitioner(GridPartitioner):
    """Prefixed UoD with Even Length Grid Partitioner"""

    def __init__(self, **kwargs):
        """
        Fixed UoD with Even Length Grid Partitioner
        """

        kwargs['preprocess'] = False

        super(GridPartitioner, self).__init__(name="Grid", **kwargs)

        self.max = kwargs.get('max', None)
        self.min = kwargs.get('min', None)

        if self.max is None or self.min is None:
            raise Exception("It is mandatory to inform the max and min parameters!")

        self.sets = self.build(None)

        self.partitions = len(self.sets)

        if self.ordered_sets is None and self.setnames is not None:
            self.ordered_sets = self.setnames[:len(self.sets)]
        else:
            self.ordered_sets = FuzzySet.set_ordered(self.sets)





