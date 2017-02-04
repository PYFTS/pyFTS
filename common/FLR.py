import numpy as np
from pyFTS.common import FuzzySet


class FLR(object):
    def __init__(self, LHS, RHS):
        self.LHS = LHS
        self.RHS = RHS

    def __str__(self):
        return self.LHS.name + " -> " + self.RHS.name


class IndexedFLR(FLR):
    def __init__(self, index, LHS, RHS):
        super(IndexedFLR, self).__init__(LHS, RHS)
        self.index = index

    def __str__(self):
        return str(self.index) + ": "+ self.LHS.name + " -> " + self.RHS.name


def generateNonRecurrentFLRs(fuzzyData):
    flrs = {}
    for i in range(2,len(fuzzyData)):
        tmp = FLR(fuzzyData[i-1],fuzzyData[i])
        flrs[str(tmp)] = tmp
    ret = [value for key, value in flrs.items()]
    return ret


def generateRecurrentFLRs(fuzzyData):
    flrs = []
    for i in np.arange(1,len(fuzzyData)):
        flrs.append(FLR(fuzzyData[i-1],fuzzyData[i]))
    return flrs


def generateIndexedFLRs(sets, indexer, data):
    flrs = []
    index = indexer.get_season_of_data(data)
    ndata = indexer.get_data(data)
    for k in np.arange(0,len(data)-1):
        lhs = FuzzySet.getMaxMembershipFuzzySet(ndata[k],sets)
        rhs = FuzzySet.getMaxMembershipFuzzySet(ndata[k+1], sets)
        season = index[k]
        flr = IndexedFLR(season,lhs,rhs)
        flrs.append(flr)
    return flrs
