"""
This module implements functions for Fuzzy Logical Relationship generation
"""

import numpy as np
from pyFTS.common import FuzzySet


class FLR(object):
    """
    Fuzzy Logical Relationship

    Represents a temporal transition of the fuzzy set LHS on time t for the fuzzy set RHS on time t+1.
    """
    def __init__(self, LHS, RHS):
        """
        Creates a Fuzzy Logical Relationship
        :param LHS: Left Hand Side fuzzy set
        :param RHS: Right Hand Side fuzzy set
        """
        self.LHS = LHS
        self.RHS = RHS

    def __str__(self):
        return str(self.LHS) + " -> " + str(self.RHS)


class IndexedFLR(FLR):
    """Season Indexed Fuzzy Logical Relationship"""
    def __init__(self, index, LHS, RHS):
        """
        Create a Season Indexed Fuzzy Logical Relationship
        :param index: seasonal index 
        :param LHS: Left Hand Side fuzzy set
        :param RHS: Right Hand Side fuzzy set
        """
        super(IndexedFLR, self).__init__(LHS, RHS)
        self.index = index

    def __str__(self):
        return str(self.index) + ": "+ str(self.LHS) + " -> " + str(self.RHS)


def generate_high_order_recurrent_flr(fuzzyData):
    """
    Create a ordered FLR set from a list of fuzzy sets with recurrence
    :param fuzzyData: ordered list of fuzzy sets
    :return: ordered list of FLR
    """
    flrs = []
    for i in np.arange(1,len(fuzzyData)):
        lhs = fuzzyData[i - 1]
        rhs = fuzzyData[i]
        if isinstance(lhs, list) and isinstance(rhs, list):
            for l in lhs:
                for r in rhs:
                    tmp = FLR(l, r)
                    flrs.append(tmp)
        else:
            tmp = FLR(lhs,rhs)
            flrs.append(tmp)
    return flrs


def generate_recurrent_flrs(fuzzyData):
    """
    Create a ordered FLR set from a list of fuzzy sets with recurrence
    :param fuzzyData: ordered list of fuzzy sets
    :return: ordered list of FLR
    """
    flrs = []
    for i in np.arange(1,len(fuzzyData)):
        lhs = [fuzzyData[i - 1]]
        rhs = [fuzzyData[i]]
        for l in np.array(lhs).flatten():
            for r in np.array(rhs).flatten():
                tmp = FLR(l, r)
                flrs.append(tmp)
    return flrs


def generate_non_recurrent_flrs(fuzzyData):
    """
    Create a ordered FLR set from a list of fuzzy sets without recurrence
    :param fuzzyData: ordered list of fuzzy sets
    :return: ordered list of FLR
    """
    flrs = generate_recurrent_flrs(fuzzyData)
    tmp = {}
    for flr in flrs: tmp[str(flr)] = flr
    ret = [value for key, value in tmp.items()]
    return ret


def generate_indexed_flrs(sets, indexer, data, transformation=None, alpha_cut=0.0):
    """
    Create a season-indexed ordered FLR set from a list of fuzzy sets with recurrence
    :param sets: fuzzy sets
    :param indexer: seasonality indexer 
    :param data: original data
    :return: ordered list of FLR 
    """
    flrs = []
    index = indexer.get_season_of_data(data)
    ndata = indexer.get_data(data)
    if transformation is not None:
        ndata = transformation.apply(ndata)
    for k in np.arange(1,len(ndata)):
        lhs = FuzzySet.fuzzyfy_series([ndata[k - 1]], sets, method='fuzzy',alpha_cut=alpha_cut)
        rhs = FuzzySet.fuzzyfy_series([ndata[k]], sets, method='fuzzy',alpha_cut=alpha_cut)
        season = index[k]
        for _l in np.array(lhs).flatten():
            for _r in np.array(rhs).flatten():
                flr = IndexedFLR(season,_l,_r)
                flrs.append(flr)
    return flrs
