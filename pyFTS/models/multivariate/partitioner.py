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

    def format_data(self, data):
        ndata = {}
        for var in self.explanatory_variables:
            ndata[var.name] = var.partitioner.extractor(data[var.data_label])

        return ndata

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

    def search(self, data, **kwargs):
        '''
        Perform a search for the nearest fuzzy sets of the point 'data'. This function were designed to work with several
        overlapped fuzzy sets.

        :param data: the value to search for the nearest fuzzy sets
        :param type: the return type: 'index' for the fuzzy set indexes or 'name' for fuzzy set names.
        :return: a list with the nearest fuzzy sets
        '''
        if self.kdtree is None:
            self.build_index()

        type = kwargs.get('type', 'index')

        ndata = [data[k.name] for k in self.explanatory_variables]
        _, ix = self.kdtree.query(ndata, self.neighbors)

        if not isinstance(ix, (list, np.ndarray)):
            ix = [ix]

        if self.optimize:
            tmp = []
            for k in ix:
                tmp.append(self.index[k])
                self.count[self.index[k]] = 1

        if type == 'name':
            return [self.index[k] for k in ix]
        elif type == 'index':
            return sorted(ix)




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
