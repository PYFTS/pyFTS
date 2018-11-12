import numpy as np
import pandas as pd
from pyFTS.common import FuzzySet, Composite

class MultivariateFuzzySet(Composite.FuzzySet):
    """
    Multivariate Composite Fuzzy Set
    """
    def __init__(self, name):
        """
        Create an empty composite fuzzy set
        :param name: fuzzy set name
        """
        super(MultivariateFuzzySet, self).__init__(name)
        self.sets = {}

    def append_set(self, variable, set):
        """
        Appends a new fuzzy set from a new variable

        :param variable: an multivariate.variable instance
        :param set: an common.FuzzySet instance
        """
        self.sets[variable] = set

    def membership(self, x):
        mv = []
        for var in self.sets.keys():
            data = x[var]
            mv.append(self.sets[var].membership(data))
        return np.nanmin(mv)



def fuzzyfy_instance(data_point, var):
    fsets = FuzzySet.fuzzyfy(data_point, var.partitioner, mode='sets', method='fuzzy', alpha_cut=var.alpha_cut)
    return [(var.name, fs) for fs in fsets]

def fuzzyfy_instance_clustered(data_point, cluster, alpha_cut=0.0):
    fsets = []
    for fset in cluster.sets:
        if cluster.sets[fset].membership(data_point) > alpha_cut:
            fsets.append(fset)
    return fsets



