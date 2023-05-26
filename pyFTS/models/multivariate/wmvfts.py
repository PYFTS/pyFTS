from pyFTS.common import fts, FuzzySet, FLR, Membership, tree, Activations
from pyFTS.partitioners import Grid
from pyFTS.models.multivariate import mvfts, FLR as MVFLR, common, flrg as mvflrg

import numpy as np
import pandas as pd


class WeightedFLRG(mvflrg.FLRG):
    """
    Weighted Multivariate Fuzzy Logical Rule Group
    """

    def __init__(self, **kwargs):
        super(WeightedFLRG, self).__init__(**kwargs)
        self.order = kwargs.get('order', 1)
        self.LHS = kwargs.get('lhs', {})
        self.RHS = {}
        self.count = 0.0
        self.w = None

    def append_rhs(self, fset, **kwargs):
        count = kwargs.get('count', 1.0)
        if fset not in self.RHS:
            self.RHS[fset] = count
        else:
            self.RHS[fset] += count
        self.count += count

    def weights(self):
        if self.w is None:
            self.w = np.array([self.RHS[c] / self.count for c in self.RHS.keys()])
        return self.w

    def get_midpoint(self, sets):
        if self.midpoint is None:
            mp = np.array([sets[c].centroid for c in self.RHS.keys()])
            self.midpoint = mp.dot(self.weights())

        return self.midpoint

    def get_lower(self, sets):
        if self.lower is None:
            lw = np.array([sets[s].lower for s in self.RHS.keys()])
            self.lower = lw.dot(self.weights())
        return self.lower

    def get_upper(self, sets):
        if self.upper is None:
            up = np.array([sets[s].upper for s in self.RHS.keys()])
            self.upper = up.dot(self.weights())
        return self.upper

    def __str__(self):
        _str = ""
        for k in self.RHS.keys():
            _str += ", " if len(_str) > 0 else ""
            _str += k + " (" + str(round( self.RHS[k] / self.count, 3)) + ")"

        return self.get_key() + " -> " + _str


class WeightedMVFTS(mvfts.MVFTS):
    """
    Weighted Multivariate FTS
    """
    def __init__(self, **kwargs):
        super(WeightedMVFTS, self).__init__(order=1, **kwargs)
        self.shortname = "WeightedMVFTS"
        self.name = "Weighted Multivariate FTS"
        self.has_classification = True
        self.class_weigths : dict = kwargs.get("class_weights", {})
        

    def generate_flrg(self, flrs):
        for flr in flrs:
            flrg = WeightedFLRG(lhs=flr.LHS)

            if flrg.get_key() not in self.flrgs:
                self.flrgs[flrg.get_key()] = flrg

            self.flrgs[flrg.get_key()].append_rhs(flr.RHS)

    def classify(self, data, **kwargs):
        if len(self.class_weights) == 0:
            self.class_weights = {k : 1.0 for k in self.target_variable.partitioner.sets.keys()}
        ret = []
        ndata = self.apply_transformations(data)
        activation = kwargs.get('activation', Activations.scale)
        for index, row in ndata.iterrows() if isinstance(ndata, pd.DataFrame) else enumerate(ndata):
            data_point = self.format_data(row)
            flrs = self.generate_lhs_flrs(data_point)
            classification = {k : 0 for k in self.target_variable.partitioner.sets.keys()}
            memberships = 0
            weights = 0
            for flr in flrs:
                flrg = mvflrg.FLRG(lhs=flr.LHS)
                if flrg.get_key() in self.flrgs:
                    _flrg = self.flrgs[flrg.get_key()]
                    mb = _flrg.get_membership(data_point, self.explanatory_variables)
                    memberships += mb
                    for k,v in _flrg.RHS.items():
                        classification[k] += (v / _flrg.count) * mb
                
            classification = activation(classification, self.class_weigths)

            ret.append(classification)

        return ret
