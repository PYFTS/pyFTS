from pyFTS.partitioners import partitioner
from pyFTS.models.multivariate.common import MultivariateFuzzySet, fuzzyfy_instance_clustered
from itertools import product
from scipy.spatial import KDTree
import numpy as np
import pandas as pd


class MultivariatePartitioner(partitioner.Partitioner):
    """
    Base class for partitioners which use the MultivariateFuzzySet
    """

    def __init__(self, **kwargs):
        super(MultivariatePartitioner, self).__init__(name="MultivariatePartitioner", preprocess=False, **kwargs)

        self.type = 'multivariate'
        self.sets = {}
        self.kdtree = None
        self.index = {}
        self.explanatory_variables = kwargs.get('explanatory_variables', [])
        self.target_variable = kwargs.get('target_variable', None)
        self.neighbors = kwargs.get('neighbors', 2)
        self.optimize = kwargs.get('optimize', True)
        if self.optimize:
            self.count = {}
        data = kwargs.get('data', None)
        self.build(data)

    def build(self, data):
        pass

    def append(self, fset):
        self.sets[fset.name] = fset

    def prune(self):

        if not self.optimize:
            return

        for fset in [fs for fs in self.sets.keys()]:
            if fset not in self.count:
                fs = self.sets.pop(fset)
                del (fs)

        self.build_index()

    def knn(self, data):
        tmp = [data[k.name]
               for k in self.explanatory_variables]
        tmp, ix = self.kdtree.query(tmp, self.neighbors)

        if not isinstance(ix, (list, np.ndarray)):
            ix = [ix]

        if self.optimize:
            tmp = []
            for k in ix:
                tmp.append(self.index[k])
                self.count[self.index[k]] = 1
            return tmp
        else:
            return [self.index[k] for k in ix]

    def fuzzyfy(self, data, **kwargs):
        return fuzzyfy_instance_clustered(data, self, **kwargs)

    def change_target_variable(self, variable):
        for fset in self.sets.values():
            fset.set_target_variable(variable)

    def build_index(self):

        midpoints = []

        self.index = {}

        for ct, fset in enumerate(self.sets.values()):
            mp = []
            for vr in self.explanatory_variables:
                mp.append(fset.sets[vr.name].centroid)
            midpoints.append(mp)
            self.index[ct] = fset.name

        import sys
        sys.setrecursionlimit(100000)

        self.kdtree = KDTree(midpoints)

        sys.setrecursionlimit(1000)
