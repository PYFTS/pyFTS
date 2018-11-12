from pyFTS.partitioners import partitioner
from pyFTS.models.multivariate.common import MultivariateFuzzySet
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
        self.build(None)

    def build(self, data):

        fsets = [[x for x in k.partitioner.sets.values()]
                 for k in self.mvfts.explanatory_variables]

        midpoints = []
        index = {}

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

        self.kdtree = KDTree(midpoints)

    def knn(self, data):
        tmp = [data[k.name] for k in self.mvfts.explanatory_variables]
        tmp, ix = self.kdtree.query(tmp,2)

        if not isinstance(ix, (list, np.ndarray)):
            ix = [ix]

        return [self.index[k] for k in ix]

