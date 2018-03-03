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


def set_ordered(fuzzySets):
    return [k for k in sorted(fuzzySets.keys())]


def fuzzyfy_instance(inst, fuzzySets, ordered_sets=None):
    """
    Calculate the membership values for a data point given fuzzy sets
    :param inst: data point
    :param fuzzySets: dict of fuzzy sets
    :return: array of membership values
    """

    if ordered_sets is None:
        ordered_sets = set_ordered(fuzzySets)

    mv = []
    for key in ordered_sets:
        mv.append( fuzzySets[key].membership(inst))
    return np.array(mv)


def fuzzyfy_instances(data, fuzzySets, ordered_sets=None):
    """
    Calculate the membership values for a data point given fuzzy sets
    :param inst: data point
    :param fuzzySets: dict of fuzzy sets
    :return: array of membership values
    """
    ret = []
    if ordered_sets is None:
        ordered_sets = set_ordered(fuzzySets)
    for inst in data:
        mv = np.array([fuzzySets[key].membership(inst) for key in ordered_sets])
        ret.append(mv)
    return ret


def get_maximum_membership_fuzzyset(inst, fuzzySets, ordered_sets=None):
    """
    Fuzzify a data point, returning the fuzzy set with maximum membership value
    :param inst: data point
    :param fuzzySets: dict of fuzzy sets
    :return: fuzzy set with maximum membership
    """
    if ordered_sets is None:
        ordered_sets = set_ordered(fuzzySets)
    mv = np.array([fuzzySets[key].membership(inst) for key in ordered_sets])
    key = ordered_sets[np.argwhere(mv == max(mv))[0, 0]]
    return fuzzySets[key]


def get_maximum_membership_fuzzyset_index(inst, fuzzySets):
    """
    Fuzzify a data point, returning the fuzzy set with maximum membership value
    :param inst: data point
    :param fuzzySets: dict of fuzzy sets
    :return: fuzzy set with maximum membership
    """
    mv = fuzzyfy_instance(inst, fuzzySets)
    return np.argwhere(mv == max(mv))[0, 0]


def fuzzyfy_series_old(data, fuzzySets, method='maximum'):
    fts = []
    for item in data:
        fts.append(get_maximum_membership_fuzzyset(item, fuzzySets).name)
    return fts


def fuzzyfy_series(data, fuzzySets, method='maximum'):
    fts = []
    ordered_sets = set_ordered(fuzzySets)
    for t, i in enumerate(data):
        mv = np.array([fuzzySets[key].membership(i) for key in ordered_sets])
        if len(mv) == 0:
            sets = check_bounds(i, fuzzySets.items(), ordered_sets)
        else:
            if method == 'fuzzy':
                ix = np.ravel(np.argwhere(mv > 0.0))
                sets = [fuzzySets[ordered_sets[i]].name for i in ix]
            elif method == 'maximum':
                mx = max(mv)
                ix = np.ravel(np.argwhere(mv == mx))
                sets = fuzzySets[ordered_sets[ix[0]]].name
        fts.append(sets)
    return fts


def check_bounds(data, sets, ordered_sets):
    if data < sets[ordered_sets[0]].get_lower():
        return sets[ordered_sets[0]]
    elif data > sets[ordered_sets[-1]].get_upper():
        return sets[ordered_sets[-1]]


def check_bounds_index(data, sets, ordered_sets):
    if data < sets[ordered_sets[0]].get_lower():
        return 0
    elif data > sets[ordered_sets[-1]].get_upper():
        return len(sets) -1
