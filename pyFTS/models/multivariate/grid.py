from pyFTS.models.multivariate import partitioner
from pyFTS.models.multivariate.common import MultivariateFuzzySet, fuzzyfy_instance_clustered
from itertools import product
from scipy.spatial import KDTree
import numpy as np
import pandas as pd


class GridCluster(partitioner.MultivariatePartitioner):
    """
    A cartesian product of all fuzzy sets of all variables
    """

    def __init__(self, **kwargs):
        super(GridCluster, self).__init__(**kwargs)
        self.name="GridCluster"
        self.build(None)

    def build(self, data):

        fsets = [[x for x in k.partitioner.sets.values()]
                 for k in self.explanatory_variables]
        c = 0
        for k in product(*fsets):
            mvfset = MultivariateFuzzySet(target_variable=self.target_variable)
            for fset in k:
                mvfset.append_set(fset.variable, fset)

            self.sets[mvfset.name] = mvfset
            c += 1

        self.build_index()

