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

    def defuzzyfy(self, values, mode='both'):
        if not isinstance(values, list):
            values = [values]

        ret = []
        for val in values:
            if mode == 'both':
                num = []
                den = []
                for fset, mv in val:
                    num.append(self.sets[fset].centroid * mv)
                    den.append(mv)
                ret.append(np.nansum(num) / np.nansum(den))
            elif mode == 'both':
                num = np.mean([self.sets[fset].centroid for fset in val])
                ret.append(num)
            elif mode == 'vector':
                num = []
                den = []
                for fset, mv in enumerate(val):
                    num.append(self.sets[self.ordered_sets[fset]].centroid * mv)
                    den.append(mv)
                ret.append(np.nansum(num) / np.nansum(den))
            else:
                raise Exception('Unknown deffuzyfication mode')

        return ret


class IncrementalGridCluster(partitioner.MultivariatePartitioner):
    """
    Create combinations of fuzzy sets of the variables on demand, incrementally increasing the
    multivariate fuzzy set base.
    """

    def __init__(self, **kwargs):
        super(IncrementalGridCluster, self).__init__(**kwargs)
        self.name="IncrementalGridCluster"
        self.build(None)

    def fuzzyfy(self, data, **kwargs):

        if isinstance(data, pd.DataFrame):
            ret = []
            for index, inst in data.iterrows():
                mv = self.fuzzyfy(inst, **kwargs)
                ret.append(mv)
            return ret

        if self.kdtree is not None:
            fsets = self.search(data, type='name')
        else:
            fsets = self.incremental_search(data, type='name')

        mode = kwargs.get('mode', 'sets')
        if mode == 'sets':
            return fsets
        elif mode == 'vector':
            raise NotImplementedError()
        elif mode == 'both':
            ret = []
            for key in fsets:
                mvfset = self.sets[key]
                ret.append((key, mvfset.membership(data)))

        return ret

    def incremental_search(self, data, **kwargs):
        alpha_cut = kwargs.get('alpha_cut', 0.)
        mode = kwargs.get('mode', 'sets')

        fsets = {}
        ret = []
        for var in self.explanatory_variables:
            ac = alpha_cut if alpha_cut > 0. else var.alpha_cut
            fsets[var.name] = var.partitioner.fuzzyfy(data[var.name], mode=mode, alpha_cut=ac)

        fsets_by_var = [fsets for var, fsets in fsets.items()]

        for p in product(*fsets_by_var):
            if mode == 'both':
                path = [fset for fset, mv in p]
                mv = [mv for fset, mv in p]
                key = ''.join(path)
            elif mode == 'sets':
                key = ''.join(p)
                path = p
            if key not in self.sets:
                mvfset = MultivariateFuzzySet(target_variable=self.target_variable)
                for ct, fs in enumerate(path):
                    mvfset.append_set(self.explanatory_variables[ct].name,
                                      self.explanatory_variables[ct].partitioner[fs])
                mvfset.name = key
                self.sets[key] = mvfset

            if mode == 'sets':
                ret.append(key)
            elif mode == 'both':
                ret.append( tuple(key,np.nanmin(mv)) )

        return ret

    def prune(self):
        self.build_index()

