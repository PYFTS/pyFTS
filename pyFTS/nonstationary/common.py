"""
Non Stationary Fuzzy Sets

GARIBALDI, Jonathan M.; JAROSZEWSKI, Marcin; MUSIKASUWAN, Salang. Nonstationary fuzzy sets.
IEEE Transactions on Fuzzy Systems, v. 16, n. 4, p. 1072-1086, 2008.
"""

import numpy as np
from pyFTS import *
from pyFTS.common import FuzzySet, Membership


class MembershipFunction(object):
    """
    Non Stationary Membership Function
    """
    def __init__(self, name, mf, parameters, **kwargs):
        """
        Non Stationary Membership Function
        :param name:
        :param mf:
        :param parameters:
        :param kwargs:
            - location: Pertubation function that affects the location of the membership function
            - location_params: Parameters for location pertubation function
            - width: Pertubation function that affects the width of the membership function
            - width_params: Parameters for width pertubation function
            - noise: Pertubation function that adds noise on the membership function
            - noise_params: Parameters for noise pertubation function
        """
        self.mf = mf
        self.parameters = parameters
        self.location = kwargs.get("location", None)
        self.location_params = kwargs.get("location_params", None)
        self.width = kwargs.get("width", None)
        self.width_params = kwargs.get("width_params", None)
        self.noise = kwargs.get("noise", None)
        self.noise_params = kwargs.get("noise_params", None)
        self.cache = {}

    def perform_location(self, t, param):
        if self.location is None:
            return param

        if not isinstance(self.location, (list,set)):
            self.location = [self.location]
            self.location_params = [self.location_params]

        l = len(self.location)

        inc = sum([self.location[k](t, self.location_params[k]) for k in np.arange(0,l)] )

        if self.mf == Membership.gaussmf:
            #changes only the mean parameter
            return [param[0] + inc, param[1]]
        elif self.mf == Membership.sigmf:
            #changes only the midpoint parameter
            return [param[0], param[1] + inc]
        elif self.mf == Membership.bellmf:
            return [param[0], param[1], param[2] + inc]
        else:
            #translate all parameters
            return [k + inc for k in param]

    def perform_width(self, t, param):
        if self.width is None:
            return param

        if not isinstance(self.width, (list, set)):
            self.width = [self.width]
            self.width_params = [self.width_params]

        l = len(self.width)

        inc = sum([self.width[k](t, self.width_params[k]) for k in np.arange(0, l)])

        if self.mf == Membership.gaussmf:
            #changes only the variance parameter
            return [param[0], param[1] + inc]
        elif self.mf == Membership.sigmf:
            #changes only the smooth parameter
            return [param[0] + inc, param[1]]
        elif self.mf == Membership.trimf:
            return [param[0] - inc, param[1], param[2] + inc]
        elif self.mf == Membership.trapmf:
            l = (param[3]-param[0])
            rab = (param[1] - param[0]) / l
            rcd = (param[3] - param[2]) / l
            return [param[0] - inc, param[1] - inc*rab, param[2] + inc*rcd, param[3] + inc]
        else:
            return param

    def membership(self, x, t):
        """
        Calculate the membership value of a given input
        :param x: input value
        :return: membership value of x at this fuzzy set
        """

        if t not in self.cache:
            param = self.parameters
            param = self.perform_location(t, param)
            param = self.perform_width(t, param)
            self.cache[t] = param

        tmp = self.mf(x, self.cache[t])

        if self.noise is not None:
            tmp += self.noise(t, self.noise_params)

        return tmp


class FuzzySet(FuzzySet.FuzzySet):
    """
    Non Stationary Fuzzy Sets
    """

    def __init__(self, name, mf, **kwargs):
        """
        Constructor
        :param name: Fuzzy Set name
        :param mf: NonStationary Membership Function
        """
        super(FuzzySet, self).__init__(name=name, mf=mf, parameters=None, centroid=None)

    def membership(self, x, t):
        return self.mf.membership(x,t)


