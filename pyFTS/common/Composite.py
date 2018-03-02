"""
Composite Fuzzy Sets
"""

import numpy as np
from pyFTS import *
from pyFTS.common import Membership, FuzzySet


class FuzzySet(FuzzySet.FuzzySet):
    """
    Composite Fuzzy Set
    """
    def __init__(self, name, superset=False):
        """
        Create an empty composite fuzzy set
        :param name: fuzzy set name
        """
        super(FuzzySet, self).__init__(name, None, None, None, type='composite')
        self.superset = superset
        if self.superset:
            self.sets = []
        else:
            self.mf = []
            self.parameters = []


    def membership(self, x):
        """
        Calculate the membership value of a given input
        :param x: input value
        :return: membership value of x at this fuzzy set
        """
        if self.superset:
            return max([s.membership(x) for s in self.sets])
        else:
            return min([self.mf[ct](x, self.parameters[ct]) for ct in np.arange(0, len(self.mf))])

    def append(self, mf, parameters):
        """
        Adds a new function to composition
        :param mf:
        :param parameters:
        :return:
        """
        self.mf.append(mf)
        self.parameters.append(parameters)

    def append_set(self, set):
        """
        Adds a new function to composition
        :param mf:
        :param parameters:
        :return:
        """
        self.sets.append(set)