import numpy as np
import math
import random as rnd
import functools, operator
from pyFTS.common import FuzzySet, Membership, Transformations


# K. H. Huarng, “Effective lengths of intervals to improve forecasting in fuzzy time series,”
# Fuzzy Sets Syst., vol. 123, no. 3, pp. 387–394, Nov. 2001.
from pyFTS.partitioners import partitioner


class HuarngPartitioner(partitioner.Partitioner):
    """Huarng Empirical Partitioner"""
    def __init__(self, data,npart,func = Membership.trimf, transformation=None, indexer=None):
        super(HuarngPartitioner, self).__init__("Huarng", data, npart, func=func, transformation=transformation, indexer=indexer)

    def build(self, data):
        diff = Transformations.Differential(1)
        data2 = diff.apply(data)
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

        dlen = self.max - self.min
        npart = math.ceil(dlen / base)
        partition = math.ceil(self.min)
        for c in range(npart):
            if self.membership_function == Membership.trimf:
                sets.append( FuzzySet.FuzzySet(self.prefix + str(c), Membership.trimf,
                                               [partition - base, partition, partition + base], partition))
            elif self.membership_function == Membership.gaussmf:
                sets.append(FuzzySet.FuzzySet(self.prefix + str(c), Membership.gaussmf,
                                              [partition, base/2], partition))
            elif self.membership_function == Membership.trapmf:
                sets.append(FuzzySet.FuzzySet(self.prefix + str(c), Membership.trapmf,
                                              [partition - base, partition - (base/2),
                                               partition + (base / 2), partition + base], partition))

            partition += base

        return sets
