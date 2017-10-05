"""
Non Stationary Fuzzy Sets

GARIBALDI, Jonathan M.; JAROSZEWSKI, Marcin; MUSIKASUWAN, Salang. Nonstationary fuzzy sets.
IEEE Transactions on Fuzzy Systems, v. 16, n. 4, p. 1072-1086, 2008.
"""

import numpy as np
from pyFTS import *
from pyFTS.common import FuzzySet as FS, Membership, FLR
from pyFTS.partitioners import partitioner
from pyFTS.nonstationary import perturbation


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
        super(FuzzySet, self).__init__(name=name, mf=mf, parameters=parameters, centroid=None)
    
        self.location = kwargs.get("location", None)
        self.location_params = kwargs.get("location_params", None)
        self.location_roots = kwargs.get("location_roots", 0)
        self.width = kwargs.get("width", None)
        self.width_params = kwargs.get("width_params", None)
        self.width_roots = kwargs.get("width_roots", 0)
        self.noise = kwargs.get("noise", None)
        self.noise_params = kwargs.get("noise_params", None)
        self.perturbated_parameters = {}
    
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
    
        tmp = self.mf(x, self.perturbated_parameters[t])
    
        if self.noise is not None:
            tmp += self.noise(t, self.noise_params)
    
        return tmp
    
    def perturbate_parameters(self, t):
        if t not in self.perturbated_parameters:
            param = self.parameters
            param = self.perform_location(t, param)
            param = self.perform_width(t, param)
            self.perturbated_parameters[t] = param
            
    def get_midpoint(self, t):

        self.perturbate_parameters(t)

        if self.mf == Membership.gaussmf:
            return self.perturbated_parameters[t][0]
        elif self.mf == Membership.sigmf:
            return self.perturbated_parameters[t][1]
        elif self.mf == Membership.trimf:
            return self.perturbated_parameters[t][1]
        elif self.mf == Membership.trapmf:
            param = self.perturbated_parameters[t]
            return (param[2] - param[1]) / 2
        else:
            return self.perturbated_parameters[t]

    def get_lower(self, t):

        self.perturbate_parameters(t)
        param = self.perturbated_parameters[t]

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
        param = self.perturbated_parameters[t]

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
            tmp += "Loc. Pert.: "
            for ct, f in enumerate(self.location):
                tmp += str(f.__name__) + "(" + str(self.location_params[ct]) + ") "
        if self.width is not None:
            tmp += "Wid. Pert.: "
            for ct, f in enumerate(self.width):
                tmp += str(f.__name__) + "(" + str(self.width_params[ct]) + ") "
        return self.name + ": " + str(self.mf.__name__) + "(" + str(self.parameters) + ") " + tmp


class PolynomialNonStationaryPartitioner(partitioner.Partitioner):
    """
    Non Stationary Universe of Discourse Partitioner
    """

    def __init__(self, data, part, **kwargs):
        """"""
        super(PolynomialNonStationaryPartitioner, self).__init__(name=part.name, data=data, npart=part.partitions,
                                                                 func=part.membership_function, names=part.setnames,
                                                                 prefix=part.prefix, transformation=part.transformation,
                                                                 indexer=part.indexer)

        self.sets = []

        loc_params, wid_params = self.get_polynomial_perturbations(data, **kwargs)

        for ct, set in enumerate(part.sets):
            loc_roots = np.roots(loc_params[ct])[0]
            wid_roots = np.roots(wid_params[ct])[0]
            tmp = FuzzySet(set.name, set.mf, set.parameters,
                           location=perturbation.polynomial,
                           location_params=loc_params[ct],
                           location_roots=loc_roots, #**kwargs)
                           width=perturbation.polynomial,
                           width_params=wid_params[ct],
                           width_roots=wid_roots, **kwargs)

            self.sets.append(tmp)

    def poly_width(self, par1, par2, rng, deg):
        a = np.polyval(par1, rng)
        b = np.polyval(par2, rng)
        diff = [b[k] - a[k] for k in rng]
        tmp = np.polyfit(rng, diff, deg=deg)
        return tmp

    def get_polynomial_perturbations(self, data, **kwargs):
        w = kwargs.get("window_size", int(len(data) / 5))
        deg = kwargs.get("degree", 2)
        xmax = [0]
        tmax = [0]
        xmin = [0]
        tmin = [0]
        lengs = [0]
        tlengs = [0]
        l = len(data)

        for i in np.arange(0, l, w):
            sample = data[i:i + w]
            tx = max(sample)
            xmax.append(tx)
            tmax.append(np.ravel(np.argwhere(data == tx)).tolist()[0])
            tn = min(sample)
            xmin.append(tn)
            tmin.append(np.ravel(np.argwhere(data == tn)).tolist()[0])
            lengs.append((tx - tn)/self.partitions)
            tlengs.append(i)


        cmax = np.polyfit(tmax, xmax, deg=deg)
        #cmax = cmax.tolist()
        cmin = np.polyfit(tmin, xmin, deg=deg)
        #cmin = cmin.tolist()


        cmed = []

        for d in np.arange(0, deg + 1):
            cmed.append(np.linspace(cmin[d], cmax[d], self.partitions)[1:self.partitions - 1])

        loc_params = [cmin.tolist()]
        for i in np.arange(0, self.partitions - 2):
            tmp = [cmed[k][i] for k in np.arange(0, deg + 1)]
            loc_params.append(tmp)
        loc_params.append(cmax.tolist())

        rng = np.arange(0, l)

        clen = []

        for i in np.arange(1, self.partitions-1):
            tmp = self.poly_width(loc_params[i - 1], loc_params[i + 1], rng, deg)
            clen.append(tmp)

        tmp = self.poly_width(loc_params[0], loc_params[1], rng, deg)
        clen.insert(0, tmp)

        tmp = self.poly_width(loc_params[self.partitions-2], loc_params[self.partitions-1], rng, deg)
        clen.append(tmp)

        tmp = (loc_params, clen)

        return tmp

    def build(self, data):
        pass


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


def fuzzySeries(data, fuzzySets):
    fts = []
    for t, i in enumerate(data):
        mv = np.array([fs.membership(i, t) for fs in fuzzySets])
        ix = np.ravel(np.argwhere(mv > 0.0))
        sets = [fuzzySets[i] for i in ix]
        fts.append(sets)
    return fts

