from pyFTS.partitioners import partitioner
from pyFTS.models.multivariate.common import MultivariateFuzzySet, fuzzyfy_instance_clustered
from itertools import product
from scipy.spatial import KDTree
import numpy as np
import pandas as pd


class GridCluster(partitioner.Partitioner):
    """
    A cartesian product of all fuzzy sets of all variables
    """

    def __init__(self, **kwargs):
        super(GridCluster, self).__init__(name="GridCluster", preprocess=False, **kwargs)

        self.mvfts = kwargs.get('mvfts', None)
        self.sets = {}
        self.kdtree = None
        self.index = {}
        self.neighbors = kwargs.get('neighbors', 2)
        self.optmize = kwargs.get('optmize', False)
        if self.optmize:
            self.count = {}
        data = kwargs.get('data', [None])
        self.build(data)

    def build(self, data):

        fsets = [[x for x in k.partitioner.sets.values()]
                 for k in self.mvfts.explanatory_variables]

        midpoints = []

        c = 0
        for k in product(*fsets):
            #key = self.prefix+str(c)
            mvfset = MultivariateFuzzySet(name="", target_variable=self.mvfts.target_variable)
            mp = []
            _key = ""
            for fset in k:
                mvfset.append_set(fset.variable, fset)
                mp.append(fset.centroid)
                _key += fset.name
            mvfset.name = _key
            self.sets[_key] = mvfset
            midpoints.append(mp)
            self.index[c] = _key
            c += 1

        import sys
        sys.setrecursionlimit(100000)

        self.kdtree = KDTree(midpoints)

        sys.setrecursionlimit(1000)

    def prune(self):

        if not self.optmize:
            return

        for fset in [fs for fs in self.sets.keys()]:
            if fset not in self.count:
                fs = self.sets.pop(fset)
                del (fs)


        vars = [k.name for k in self.mvfts.explanatory_variables]

        midpoints = []

        self.index = {}

        for ct, fset in enumerate(self.sets.values()):
            mp = []
            for vr in vars:
                mp.append(fset.sets[vr].centroid)
            midpoints.append(mp)
            self.index[ct] = fset.name

        import sys
        sys.setrecursionlimit(100000)

        self.kdtree = KDTree(midpoints)

        sys.setrecursionlimit(1000)


    def knn(self, data):
        tmp = [data[k.name]
               for k in self.mvfts.explanatory_variables]
        tmp, ix = self.kdtree.query(tmp, self.neighbors)

        if not isinstance(ix, (list, np.ndarray)):
            ix = [ix]

        if self.optmize:
            tmp = []
            for k in ix:
                tmp.append(self.index[k])
                self.count[self.index[k]] = 1
            return tmp
        else:
            return [self.index[k] for k in ix]

    def fuzzyfy(self, data, **kwargs):
        return fuzzyfy_instance_clustered(data, self, **kwargs)
