import numpy as np
import pandas as pd
from pyFTS.common import FuzzySet, Composite


class MultivariateFuzzySet(Composite.FuzzySet):
    """
    Multivariate Composite Fuzzy Set
    """
    def __init__(self, **kwargs):
        """
        Create an empty composite fuzzy set
        :param name: fuzzy set name
        """
        super(MultivariateFuzzySet, self).__init__("")
        self.sets = {}
        self.target_variable = kwargs.get('target_variable',None)

    def append_set(self, variable, set):
        """
        Appends a new fuzzy set from a new variable

        :param variable: an multivariate.variable instance
        :param set: an common.FuzzySet instance
        """
        self.sets[variable] = set

        if variable == self.target_variable.name:
            self.centroid = set.centroid

        self.name += set.name

    def set_target_variable(self, variable):
        self.target_variable = variable
        self.centroid = self.sets[variable.name].centroid

    def membership(self, x):
        mv = []
        for var in self.sets.keys():
            data = x[var]
            mv.append(self.sets[var].membership(data))
        return np.nanmin(mv)


def fuzzyfy_instance(data_point, var, tuples=True):
    fsets = var.partitioner.fuzzyfy(data_point, mode='sets', method='fuzzy', alpha_cut=var.alpha_cut)
    if tuples:
        return [(var.name, fs) for fs in fsets]
    else:
        return fsets


def fuzzyfy_instance_clustered(data_point, cluster, **kwargs):
    alpha_cut = kwargs.get('alpha_cut', 0.0)
    mode = kwargs.get('mode', 'sets')
    fsets = []
    for fset in cluster.knn(data_point):
        if cluster.sets[fset].membership(data_point) > alpha_cut:
            if mode == 'sets':
                fsets.append(fset)
            elif mode =='both':
                fsets.append( (fset, cluster.sets[fset].membership(data_point)) )
    return fsets



