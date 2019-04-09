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
        """
        self.name = name
        """The fuzzy set name"""
        self.mf = mf
        """The membership function"""
        self.parameters = parameters
        """The parameters of the membership function"""
        self.centroid = centroid
        """The fuzzy set center of mass (or midpoint)"""
        self.alpha = alpha
        """The alpha cut value"""
        self.type = kwargs.get('type', 'common')
        """The fuzzy set type (common, composite, nonstationary, etc)"""
        self.variable = kwargs.get('variable', None)
        """In multivariate time series, indicate for which variable this fuzzy set belogs"""
        self.Z = None
        """Partition function in respect to the membership function"""

        if parameters is not None:
            if self.mf == Membership.gaussmf:
                self.lower = parameters[0] - parameters[1] * 3
                self.upper = parameters[0] + parameters[1] * 3
            elif self.mf == Membership.sigmf:
                k = (parameters[1] / (2 * parameters[0]))
                self.lower = parameters[1] - k
                self.upper = parameters[1] + k
            else:
                self.lower = min(parameters)
                self.upper = max(parameters)

        self.metadata = {}

    def transform(self, x):
        """
        Preprocess the data point for non native types

        :param x:
        :return: return a native type value for the structured type
        """

        return x

    def membership(self, x):
        """
        Calculate the membership value of a given input

        :param x: input value 
        :return: membership value of x at this fuzzy set
        """
        return self.mf(self.transform(x), self.parameters) * self.alpha

    def partition_function(self, uod=None, nbins=100):
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


def __binary_search(x, fuzzy_sets, ordered_sets):
    """
    Search for elegible fuzzy sets to fuzzyfy x

    :param x: input value to be fuzzyfied
    :param fuzzy_sets:  a dictionary where the key is the fuzzy set name and the value is the fuzzy set object.
    :param ordered_sets: a list with the fuzzy sets names ordered by their centroids.
    :return: A list with the best fuzzy sets that may contain x
    """
    max_len = len(fuzzy_sets) - 1
    first = 0
    last = max_len

    while first <= last:
        midpoint = (first + last) // 2

        fs = ordered_sets[midpoint]
        fs1 = ordered_sets[midpoint - 1] if midpoint > 0 else ordered_sets[0]
        fs2 = ordered_sets[midpoint + 1] if midpoint < max_len else ordered_sets[max_len]

        if fuzzy_sets[fs1].centroid <= fuzzy_sets[fs].transform(x) <= fuzzy_sets[fs2].centroid:
            return (midpoint - 1, midpoint, midpoint + 1)
        elif midpoint <= 1:
            return [0]
        elif midpoint >= max_len:
            return [max_len]
        else:
            if fuzzy_sets[fs].transform(x) < fuzzy_sets[fs].centroid:
                last = midpoint - 1
            else:
                first = midpoint + 1


def fuzzyfy(data, partitioner, **kwargs):
    """
    A general method for fuzzyfication.

    :param data: input value to be fuzzyfied
    :param partitioner: a trained pyFTS.partitioners.Partitioner object
    :param kwargs: dict, optional arguments
    :keyword alpha_cut: the minimal membership value to be considered on fuzzyfication (only for mode='sets')
    :keyword method: the fuzzyfication method (fuzzy: all fuzzy memberships, maximum: only the maximum membership)
    :keyword mode: the fuzzyfication mode (sets: return the fuzzy sets names, vector: return a vector with the membership
    values for all fuzzy sets, both: return a list with tuples (fuzzy set, membership value) )
    :returns a list with the fuzzyfied values, depending on the mode
    
    """
    alpha_cut = kwargs.get('alpha_cut', 0.)
    mode = kwargs.get('mode', 'sets')
    method = kwargs.get('method', 'fuzzy')
    if isinstance(data, (list, np.ndarray)):
        if mode == 'vector':
            return fuzzyfy_instances(data, partitioner.sets, partitioner.ordered_sets)
        elif mode == 'both':
            mvs = fuzzyfy_instances(data, partitioner.sets, partitioner.ordered_sets)
            fs = []
            for mv in mvs:
                fsets = [(partitioner.ordered_sets[ix], mv[ix])
                         for ix in np.arange(len(mv))
                         if mv[ix] >= alpha_cut]
                fs.append(fsets)
            return fs
        else:
            return fuzzyfy_series(data, partitioner.sets, method, alpha_cut, partitioner.ordered_sets)
    else:
        if mode == 'vector':
            return fuzzyfy_instance(data, partitioner.sets, partitioner.ordered_sets)
        elif mode == 'both':
            mv = fuzzyfy_instance(data, partitioner.sets, partitioner.ordered_sets)
            fsets = [(partitioner.ordered_sets[ix], mv[ix])
                     for ix in np.arange(len(mv))
                     if mv[ix] >= alpha_cut]
            return fsets
        else:
            return get_fuzzysets(data, partitioner.sets, partitioner.ordered_sets, alpha_cut)


def set_ordered(fuzzy_sets):
    """
    Order a fuzzy set list by their centroids

    :param fuzzy_sets: a dictionary where the key is the fuzzy set name and the value is the fuzzy set object.
    :return: a list with the fuzzy sets names ordered by their centroids.
    """
    if len(fuzzy_sets) > 0:
        tmp1 = [fuzzy_sets[k] for k in fuzzy_sets.keys()]
        return [k.name for k in sorted(tmp1, key=lambda x: x.centroid)]


def fuzzyfy_instance(inst, fuzzy_sets, ordered_sets=None):
    """
    Calculate the membership values for a data point given fuzzy sets

    :param inst: data point
    :param fuzzy_sets: a dictionary where the key is the fuzzy set name and the value is the fuzzy set object.
    :param ordered_sets: a list with the fuzzy sets names ordered by their centroids.
    :return: array of membership values
    """

    if ordered_sets is None:
        ordered_sets = set_ordered(fuzzy_sets)

    mv = np.zeros(len(fuzzy_sets))

    for ix in __binary_search(inst, fuzzy_sets, ordered_sets):
        mv[ix] = fuzzy_sets[ordered_sets[ix]].membership(inst)

    return mv


def fuzzyfy_instances(data, fuzzy_sets, ordered_sets=None):
    """
    Calculate the membership values for a data point given fuzzy sets

    :param inst: data point
    :param fuzzy_sets: a dictionary where the key is the fuzzy set name and the value is the fuzzy set object.
    :param ordered_sets: a list with the fuzzy sets names ordered by their centroids.
    :return: array of membership values
    """
    ret = []
    if ordered_sets is None:
        ordered_sets = set_ordered(fuzzy_sets)
    for inst in data:
        mv = fuzzyfy_instance(inst, fuzzy_sets, ordered_sets)
        ret.append(mv)
    return ret


def get_fuzzysets(inst, fuzzy_sets, ordered_sets=None, alpha_cut=0.0):
    """
    Return the fuzzy sets which membership value for a inst is greater than the alpha_cut

    :param inst: data point
    :param fuzzy_sets:  a dictionary where the key is the fuzzy set name and the value is the fuzzy set object.
    :param ordered_sets: a list with the fuzzy sets names ordered by their centroids.
    :param alpha_cut: Minimal membership to be considered on fuzzyfication process
    :return: array of membership values
    """

    if ordered_sets is None:
        ordered_sets = set_ordered(fuzzy_sets)

    try:
        fs = [ordered_sets[ix]
              for ix in __binary_search(inst, fuzzy_sets, ordered_sets)
              if fuzzy_sets[ordered_sets[ix]].membership(inst) > alpha_cut]
        return fs
    except Exception as ex:
        raise ex


def get_maximum_membership_fuzzyset(inst, fuzzy_sets, ordered_sets=None):
    """
    Fuzzify a data point, returning the fuzzy set with maximum membership value

    :param inst: data point
    :param fuzzy_sets:  a dictionary where the key is the fuzzy set name and the value is the fuzzy set object.
    :param ordered_sets: a list with the fuzzy sets names ordered by their centroids.
    :return: fuzzy set with maximum membership
    """
    if ordered_sets is None:
        ordered_sets = set_ordered(fuzzy_sets)
    mv = np.array([fuzzy_sets[key].membership(inst) for key in ordered_sets])
    key = ordered_sets[np.argwhere(mv == max(mv))[0, 0]]
    return fuzzy_sets[key]


def get_maximum_membership_fuzzyset_index(inst, fuzzy_sets):
    """
    Fuzzify a data point, returning the fuzzy set with maximum membership value

    :param inst: data point
    :param fuzzy_sets: dict of fuzzy sets
    :return: fuzzy set with maximum membership
    """
    mv = fuzzyfy_instance(inst, fuzzy_sets)
    return np.argwhere(mv == max(mv))[0, 0]


def fuzzyfy_series_old(data, fuzzy_sets, method='maximum'):
    fts = []
    for item in data:
        fts.append(get_maximum_membership_fuzzyset(item, fuzzy_sets).name)
    return fts


def fuzzyfy_series(data, fuzzy_sets, method='maximum', alpha_cut=0.0, ordered_sets=None):
    fts = []
    if ordered_sets is None:
        ordered_sets = set_ordered(fuzzy_sets)
    for t, i in enumerate(data):
        mv = fuzzyfy_instance(i, fuzzy_sets, ordered_sets)
        if len(mv) == 0:
            sets = check_bounds(i, fuzzy_sets.items(), ordered_sets)
        else:
            if method == 'fuzzy':
                ix = np.ravel(np.argwhere(mv > alpha_cut))
                sets = [fuzzy_sets[ordered_sets[i]].name for i in ix]
            elif method == 'maximum':
                mx = max(mv)
                ix = np.ravel(np.argwhere(mv == mx))
                sets = fuzzy_sets[ordered_sets[ix[0]]].name
        fts.append(sets)
    return fts


def grant_bounds(data, fuzzy_sets, ordered_sets):
    if data < fuzzy_sets[ordered_sets[0]].lower:
        return fuzzy_sets[ordered_sets[0]].lower
    elif data > fuzzy_sets[ordered_sets[-1]].upper:
        return fuzzy_sets[ordered_sets[-1]].upper
    else:
        return data


def check_bounds(data, fuzzy_sets, ordered_sets):
    if data < fuzzy_sets[ordered_sets[0]].lower:
        return fuzzy_sets[ordered_sets[0]]
    elif data > fuzzy_sets[ordered_sets[-1]].upper:
        return fuzzy_sets[ordered_sets[-1]]


def check_bounds_index(data, fuzzy_sets, ordered_sets):
    if data < fuzzy_sets[ordered_sets[0]].get_lower():
        return 0
    elif data > fuzzy_sets[ordered_sets[-1]].get_upper():
        return len(fuzzy_sets) - 1
