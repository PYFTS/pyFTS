"""Even Length Grid Partitioner"""

import numpy as np
import math
import random as rnd
import functools, operator
from pyFTS.common import FuzzySet, Membership
from pyFTS.partitioners import partitioner


class SingletonPartitioner(partitioner.Partitioner):
    """Singleton Partitioner: Create singleton fuzzy sets for each distinct value in UoD"""

    def __init__(self, **kwargs):
        """
        Singleton Partitioner
        """
        super(SingletonPartitioner, self).__init__(name="Singleton", **kwargs)

    def build(self, data : list):
        sets = {}

        kwargs = {'type': self.type, 'variable': self.variable}

        for count, instance in enumerate(set(data)):
            _name = self.get_name(count)
            sets[_name] = FuzzySet.FuzzySet(_name, Membership.singleton, [instance], instance, **kwargs)

        return sets
