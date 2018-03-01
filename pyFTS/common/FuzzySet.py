import numpy as np
from pyFTS import *
from pyFTS.common import Membership


class FuzzySet(object):
    """
    Fuzzy Set
    """
    def __init__(self, name, mf, parameters, centroid, alpha=1.0, **kwargs):
        """
        Create a Fuzzy Set 
        :param name: fuzzy set name
        :param mf: membership function
        :param parameters: parameters of the membership function
        :param centroid: fuzzy set center of mass
        """
        self.name = name
        self.mf = mf
        self.parameters = parameters
        self.centroid = centroid
        self.alpha = alpha
        self.type = kwargs.get('type', 'common')
        self.variable = kwargs.get('variable',None)
        ":param Z: Partition function in respect to the membership function"
        self.Z = None
        if self.mf == Membership.trimf:
            self.lower = min(parameters)
            self.upper = max(parameters)
        elif self.mf == Membership.gaussmf:
            self.lower = parameters[0] - parameters[1]*3
            self.upper = parameters[0] + parameters[1]*3
        self.metadata = {}

    def membership(self, x):
        """
        Calculate the membership value of a given input
        :param x: input value 
        :return: membership value of x at this fuzzy set
        """
        return self.mf(x, self.parameters) * self.alpha

    def partition_function(self,uod=None, nbins=100):
        """
        Calculate the partition function over the membership function.
        :param uod:
        :param nbins:
        :return:
        """
        if self.Z is None and uod is not None:
            self.Z = 0.0
            for k in np.linspace(uod[0], uod[1], nbins):
                self.Z += self.membership(k)

        return self.Z

    def __str__(self):
        return self.name + ": " + str(self.mf.__name__) + "(" + str(self.parameters) + ")"


def fuzzyfy_instance(inst, fuzzySets):
    """
    Calculate the membership values for a data point given fuzzy sets
    :param inst: data point
    :param fuzzySets: list of fuzzy sets
    :return: array of membership values
    """
    mv = np.array([fs.membership(inst) for fs in fuzzySets])
    return mv


def fuzzyfy_instances(data, fuzzySets):
    """
    Calculate the membership values for a data point given fuzzy sets
    :param inst: data point
    :param fuzzySets: list of fuzzy sets
    :return: array of membership values
    """
    ret = []
    for inst in data:
        mv = np.array([fs.membership(inst) for fs in fuzzySets])
        ret.append(mv)
    return ret


def get_maximum_membership_fuzzyset(inst, fuzzySets):
    """
    Fuzzify a data point, returning the fuzzy set with maximum membership value
    :param inst: data point
    :param fuzzySets: list of fuzzy sets 
    :return: fuzzy set with maximum membership
    """
    mv = fuzzyfy_instance(inst, fuzzySets)
    return fuzzySets[np.argwhere(mv == max(mv))[0, 0]]


def get_maximum_membership_fuzzyset_index(inst, fuzzySets):
    """
    Fuzzify a data point, returning the fuzzy set with maximum membership value
    :param inst: data point
    :param fuzzySets: list of fuzzy sets 
    :return: fuzzy set with maximum membership
    """
    mv = fuzzyfy_instance(inst, fuzzySets)
    return np.argwhere(mv == max(mv))[0, 0]


def fuzzyfy_series_old(data, fuzzySets, method='maximum'):
    fts = []
    for item in data:
        fts.append(get_maximum_membership_fuzzyset(item, fuzzySets))
    return fts


def fuzzify_series(data, fuzzySets, method='maximum'):
    fts = []
    for t, i in enumerate(data):
        mv = np.array([fs.membership(i) for fs in fuzzySets])
        if len(mv) == 0:
            sets = check_bounds(i, fuzzySets)
        else:
            if method == 'fuzzy':
                ix = np.ravel(np.argwhere(mv > 0.0))
                sets = [fuzzySets[i] for i in ix]
            elif method == 'maximum':
                mx = max(mv)
                ix = np.ravel(np.argwhere(mv == mx))
                sets = fuzzySets[ix[0]]
        fts.append(sets)
    return fts


def check_bounds(data, sets):
    if data < sets[0].get_lower():
        return sets[0]
    elif data > sets[-1].get_upper():
        return sets[-1]


def check_bounds_index(data, sets):
    if data < sets[0].get_lower():
        return 0
    elif data > sets[-1].get_upper():
        return len(sets) -1
