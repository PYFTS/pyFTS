"""Simple Partitioner for manually informed fuzzy sets"""

import numpy as np
import math
import random as rnd
import functools, operator
from pyFTS.common import FuzzySet, Membership
from pyFTS.partitioners import partitioner


class SimplePartitioner(partitioner.Partitioner):
    """Simple Partitioner for manually informed fuzzy sets"""

    def __init__(self, **kwargs):
        """
        Simple Partitioner - the fuzzy sets are informed manually
        """
        kwargs['preprocess'] = False

        super(SimplePartitioner, self).__init__(name="Simple", **kwargs)

        self.partitions = 0

    def append(self, name, mf, parameters, **kwargs):
        """
        Append a new partition (fuzzy set) to the partitioner

        :param name: Fuzzy set name
        :param mf: One of the pyFTS.common.Membership functions
        :param parameters: A list with the parameters for the membership function
        :param kwargs: Optional arguments for the fuzzy set
        """
        if name is None or len(name) == 0:
            raise ValueError("The name of the fuzzy set cannot be empty")

        if name in self.sets:
            raise ValueError("This name has already been used")

        if mf is None or mf not in (Membership.trimf, Membership.gaussmf,
                                    Membership.trapmf, Membership.singleton,
                                    Membership.sigmf):
            raise ValueError("The mf parameter should be one of pyFTS.common.Membership functions")

        if mf == Membership.trimf:
            if len(parameters) != 3:
                raise ValueError("Incorrect number of parameters for the Membership.trimf")

            centroid = parameters[1]
        elif mf == Membership.gaussmf:
            if len(parameters) != 2:
                raise ValueError("Incorrect number of parameters for the Membership.gaussmf")

            centroid = parameters[0]
        elif mf == Membership.trapmf:
            if len(parameters) != 4:
                raise ValueError("Incorrect number of parameters for the Membership.trapmf")

            centroid = (parameters[1]+parameters[2])/2
        elif mf == Membership.singleton:
            if len(parameters) != 1:
                raise ValueError("Incorrect number of parameters for the Membership.singleton")

            centroid = parameters[0]
        elif mf == Membership.sigmf:
            if len(parameters) != 2:
                raise ValueError("Incorrect number of parameters for the Membership.sigmf")

            centroid = parameters[1] + (parameters[1] / (2 * parameters[0]))

        self.sets[name] = FuzzySet.FuzzySet(name, mf, parameters, centroid, **kwargs)
        self.partitions += 1

        self.ordered_sets = [key for key in sorted(self.sets.keys(), key=lambda k: self.sets[k].centroid)]

        self.min = self.sets[self.ordered_sets[0]].lower
        self.max = self.sets[self.ordered_sets[-1]].upper


