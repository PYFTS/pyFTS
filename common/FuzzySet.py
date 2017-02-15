import numpy as np
from pyFTS import *
from pyFTS.common import Membership


class FuzzySet:
    def __init__(self, name, mf, parameters, centroid):
        self.name = name
        self.mf = mf
        self.parameters = parameters
        self.centroid = centroid
        if self.mf == Membership.trimf:
            self.lower = min(parameters)
            self.upper = max(parameters)
        elif self.mf == Membership.gaussmf:
            self.lower = parameters[0] - parameters[1]*3
            self.upper = parameters[0] + parameters[1]*3

    def membership(self, x):
        return self.mf(x, self.parameters)

    def __str__(self):
        return self.name + ": " + str(self.mf.__name__) + "(" + str(self.parameters) + ")"


def fuzzyInstance(inst, fuzzySets):
    mv = np.array([fs.membership(inst) for fs in fuzzySets])
    return mv


def getMaxMembershipFuzzySet(inst, fuzzySets):
    mv = fuzzyInstance(inst, fuzzySets)
    return fuzzySets[np.argwhere(mv == max(mv))[0, 0]]

def getMaxMembershipFuzzySetIndex(inst, fuzzySets):
    mv = fuzzyInstance(inst, fuzzySets)
    return np.argwhere(mv == max(mv))[0, 0]


def fuzzySeries(data, fuzzySets):
    fts = []
    for item in data:
        fts.append(getMaxMembershipFuzzySet(item, fuzzySets))
    return fts
