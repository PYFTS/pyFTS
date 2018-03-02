"""
Non Stationary Fuzzy Sets

GARIBALDI, Jonathan M.; JAROSZEWSKI, Marcin; MUSIKASUWAN, Salang. Nonstationary fuzzy sets.
IEEE Transactions on Fuzzy Systems, v. 16, n. 4, p. 1072-1086, 2008.
"""

import numpy as np
from pyFTS import *
from pyFTS.common import FuzzySet as FS, Membership, FLR
from pyFTS.partitioners import partitioner
from pyFTS.models.nonstationary import perturbation


class FuzzySet(FS.FuzzySet):
    """
    Non Stationary Fuzzy Sets
    """

    def __init__(self, name, mf, parameters, **kwargs):
        """
        Constructor
        :param name:
        :param mf: Fuzzy Membership Function
        :param parameters:
        :param kwargs:
            - location: Pertubation function that affects the location of the membership function
            - location_params: Parameters for location pertubation function
            - width: Pertubation function that affects the width of the membership function
            - width_params: Parameters for width pertubation function
            - noise: Pertubation function that adds noise on the membership function
            - noise_params: Parameters for noise pertubation function
        """
        super(FuzzySet, self).__init__(name=name, mf=mf, parameters=parameters, centroid=None, alpha=1.0, **kwargs)
    
        self.location = kwargs.get("location", None)
        self.location_params = kwargs.get("location_params", None)
        self.location_roots = kwargs.get("location_roots", 0)
        self.width = kwargs.get("width", None)
        self.width_params = kwargs.get("width_params", None)
        self.width_roots = kwargs.get("width_roots", 0)
        self.noise = kwargs.get("noise", None)
        self.noise_params = kwargs.get("noise_params", None)
        self.perturbated_parameters = {}
        self.type = 'nonstationary'
    
        if self.location is not None and not isinstance(self.location, (list, set)):
            self.location = [self.location]
            self.location_params = [self.location_params]
            self.location_roots = [self.location_roots]
    
        if self.width is not None and not isinstance(self.width, (list, set)):
            self.width = [self.width]
            self.width_params = [self.width_params]
            self.width_roots = [self.width_roots]
    
    def perform_location(self, t, param):
        if self.location is None:
            return param
    
        l = len(self.location)
    
        inc = sum([self.location[k](t + self.location_roots[k], self.location_params[k]) for k in np.arange(0, l)])
    
        if self.mf == Membership.gaussmf:
            # changes only the mean parameter
            return [param[0] + inc, param[1]]
        elif self.mf == Membership.sigmf:
            # changes only the midpoint parameter
            return [param[0], param[1] + inc]
        elif self.mf == Membership.bellmf:
            return [param[0], param[1], param[2] + inc]
        else:
            # translate all parameters
            return [k + inc for k in param]
    
    def perform_width(self, t, param):
        if self.width is None:
            return param
    
        l = len(self.width)
    
        inc = sum([self.width[k](t + self.width_roots[k], self.width_params[k]) for k in np.arange(0, l)])
    
        if self.mf == Membership.gaussmf:
            # changes only the variance parameter
            return [param[0], param[1] + inc]
        elif self.mf == Membership.sigmf:
            # changes only the smooth parameter
            return [param[0] + inc, param[1]]
        elif self.mf == Membership.trimf:
            tmp = inc / 2
            return [param[0] - tmp, param[1], param[2] + tmp]
        elif self.mf == Membership.trapmf:
            l = (param[3] - param[0])
            rab = (param[1] - param[0]) / l
            rcd = (param[3] - param[2]) / l
            return [param[0] - inc, param[1] - inc * rab, param[2] + inc * rcd, param[3] + inc]
        else:
            return param
    
    def membership(self, x, t):
        """
        Calculate the membership value of a given input
        :param x: input value
        :return: membership value of x at this fuzzy set
        """
    
        self.perturbate_parameters(t)
    
        tmp = self.mf(x, self.perturbated_parameters[str(t)])
    
        if self.noise is not None:
            tmp += self.noise(t, self.noise_params)
    
        return tmp
    
    def perturbate_parameters(self, t):
        if str(t) not in self.perturbated_parameters:
            param = self.parameters
            if isinstance(t, (list, set)):
                param = self.perform_location(t[0], param)
                param = self.perform_width(t[1], param)
            else:
                param = self.perform_location(t, param)
                param = self.perform_width(t, param)
            self.perturbated_parameters[str(t)] = param
            
    def get_midpoint(self, t):

        self.perturbate_parameters(t)
        param = self.perturbated_parameters[str(t)]

        if self.mf == Membership.gaussmf:
            return param[0]
        elif self.mf == Membership.sigmf:
            return param[1]
        elif self.mf == Membership.trimf:
            return param[1]
        elif self.mf == Membership.trapmf:
            return (param[2] - param[1]) / 2
        else:
            return param

    def get_lower(self, t):

        self.perturbate_parameters(t)
        param = self.perturbated_parameters[str(t)]

        if self.mf == Membership.gaussmf:
            return param[0] - 3*param[1]
        elif self.mf == Membership.sigmf:
            return param[0] - param[1]
        elif self.mf == Membership.trimf:
            return param[0]
        elif self.mf == Membership.trapmf:
            return param[0]
        else:
            return param

    def get_upper(self, t):

        self.perturbate_parameters(t)
        param = self.perturbated_parameters[str(t)]

        if self.mf == Membership.gaussmf:
            return param[0] + 3*param[1]
        elif self.mf == Membership.sigmf:
            return param[0] + param[1]
        elif self.mf == Membership.trimf:
            return param[2]
        elif self.mf == Membership.trapmf:
            return param[3]
        else:
            return param

    def __str__(self):
        tmp = ""
        if self.location is not None:
            tmp += "Location: "
            for ct, f in enumerate(self.location):
                tmp += str(f.__name__) + "(" + str(["{0:.2f}".format(p) for p in self.location_params[ct]]) + ") "
        if self.width is not None:
            tmp += "Width: "
            for ct, f in enumerate(self.width):
                tmp += str(f.__name__) + "(" + str(["{0:.2f}".format(p) for p in self.width_params[ct]]) + ") "
        tmp = "(" + str(["{0:.2f}".format(p) for p in self.parameters]) + ") " + tmp
        return self.name + ": " + str(self.mf.__name__) + tmp


def fuzzify(inst, t, fuzzySets):
    """
    Calculate the membership values for a data point given nonstationary fuzzy sets
    :param inst: data points
    :param t: time displacement of the instance
    :param fuzzySets: list of fuzzy sets
    :return: array of membership values
    """
    ret = []
    if not isinstance(inst, list):
        inst = [inst]
    for t, i in enumerate(inst):
        mv = np.array([fs.membership(i, t) for fs in fuzzySets])
        ret.append(mv)
    return ret


def fuzzySeries(data, fuzzySets, window_size=1, method='fuzzy', const_t= None):
    fts = []
    for t, i in enumerate(data):
        tdisp = window_index(t, window_size) if const_t is None else const_t
        mv = np.array([fs.membership(i, tdisp) for fs in fuzzySets])
        if len(mv) == 0:
            sets = [check_bounds(i, fuzzySets, tdisp)]
        else:
            if method == 'fuzzy':
                ix = np.ravel(np.argwhere(mv > 0.0))
            elif method == 'maximum':
                mx = max(mv)
                ix = np.ravel(np.argwhere(mv == mx))
            sets = [fuzzySets[i] for i in ix]
        fts.append(sets)
    return fts


def window_index(t, window_size):
    if isinstance(t, (list, set)):
        return t
    return t - (t % window_size)


def check_bounds(data, sets, t):
    if data < sets[0].get_lower(t):
        return sets[0]
    elif data > sets[-1].get_upper(t):
        return sets[-1]


def check_bounds_index(data, sets, t):
    if data < sets[0].get_lower(t):
        return 0
    elif data > sets[-1].get_upper(t):
        return len(sets) -1
