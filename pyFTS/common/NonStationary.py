import numpy as np
from pyFTS import *
from pyFTS.common import FuzzySet, Membership


class NonStationaryFuzzySet(FuzzySet.FuzzySet):
    """
    Non Stationary Fuzzy Sets

    GARIBALDI, Jonathan M.; JAROSZEWSKI, Marcin; MUSIKASUWAN, Salang. Nonstationary fuzzy sets.
    IEEE Transactions on Fuzzy Systems, v. 16, n. 4, p. 1072-1086, 2008.
    """

    def __init__(self, name, mf, parameters):
        """
        Constructor
        :param name: Fuzzy Set name
        :param mf: Membership Function
        :param pf: Pertubation Function
        :param parameters: initial parameters of the membership function
        :param pf_parameters: parameters of the membership pertubation function
        """
        super(FuzzySet, self).__init__(order=1, name=name, **kwargs)
        self.name = name
        self.mf = mf
        self.parameters = parameters
        self.pf = []
        self.pf_parameters = []

    def appendPertubation(self, pf, pf_parameters):
        """
        Append a pertubation function to the non-stationary fuzzy set
        :param pf:
        :param pf_parameters:
        :return:
        """
        self.pf.append(pf)
        self.pf_parameters.append(pf_parameters)
