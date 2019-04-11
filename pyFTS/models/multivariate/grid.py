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


class IncrementalGridCluster(partitioner.MultivariatePartitioner):
    def __init__(self, **kwargs):
        super(IncrementalGridCluster, self).__init__(**kwargs)
        self.name="IncrementalGridCluster"
        self.build(None)

    def fuzzyfy(self, data, **kwargs):

        if isinstance(data, pd.DataFrame):
            ret = []
            for inst in data.iterrows():
                mv = self.fuzzyfy(inst, **kwargs)
                ret.append(mv)
            return ret

        alpha_cut = kwargs.get('alpha_cut', 0.)
        mode = kwargs.get('mode', 'sets')

        fsets = {}
        ret = []
        for var in self.explanatory_variables:
            fsets[var.name] = var.partitioner.fuzzyfy(data[var.name], mode='sets')

        fset = [val for key, val in fsets.items()]

        for p in product(*fset):
            key = ''.join(p)
            if key not in self.sets:
                mvfset = MultivariateFuzzySet(target_variable=self.target_variable)
                for ct, fs in enumerate(p):
                    mvfset.append_set(self.explanatory_variables[ct].name,
                                      self.explanatory_variables[ct].partitioner[fs])
                mvfset.name = key
                self.sets[key] = mvfset

            if mode=='sets':
                ret.append(key)
            elif mode=='vector':
                raise NotImplementedError()
            elif mode == 'both':
                mvfset = self.sets[key]
                ret.append((key, mvfset.membership(data)))

        return ret

    def prune(self):
        pass

