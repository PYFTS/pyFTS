from pyFTS.common import fts, FuzzySet, FLR, Membership, tree
from pyFTS.partitioners import Grid
from pyFTS.models.multivariate import FLR as MVFLR, common, flrg as mvflrg

import numpy as np
import pandas as pd


class MVFTS(fts.FTS):
    """
    Multivariate extension of Chen's ConventionalFTS method
    """
    def __init__(self, name, **kwargs):
        super(MVFTS, self).__init__(1, name, **kwargs)
        self.explanatory_variables = []
        self.target_variable = None
        self.flrgs = {}
        self.is_multivariate = True

    def append_variable(self, var):
        self.explanatory_variables.append(var)

    def format_data(self, data):
        ndata = {}
        for var in self.explanatory_variables:
            ndata[var.name] = data[var.data_label]

        return ndata

    def apply_transformations(self, data, params=None, updateUoD=False, **kwargs):
        ndata = data.copy(deep=True)
        for var in self.explanatory_variables:
            ndata[var.data_label] = var.apply_transformations(data[var.data_label].values)

        return ndata

    def generate_lhs_flrs(self, data):
        flrs = []
        lags = {}
        for vc, var in enumerate(self.explanatory_variables):
            data_point = data[var.data_label]
            lags[vc] = common.fuzzyfy_instance(data_point, var)

        root = tree.FLRGTreeNode(None)

        tree.build_tree_without_order(root, lags, 0)

        for p in root.paths():
            path = list(reversed(list(filter(None.__ne__, p))))

            flr = MVFLR.FLR()

            for v, s in path:
                flr.set_lhs(v, s)

            if len(flr.LHS.keys()) == len(self.explanatory_variables):
                flrs.append(flr)

        return flrs

    def generate_flrs(self, data):
        flrs = []
        for ct in range(1, len(data.index)):
            ix = data.index[ct-1]
            data_point = data.loc[ix]

            tmp_flrs = self.generate_lhs_flrs(data_point)

            target_ix = data.index[ct]
            target_point = data[self.target_variable.data_label][target_ix]
            target = common.fuzzyfy_instance(target_point, self.target_variable)

            for flr in tmp_flrs:
                for v, s in target:
                    flr.set_rhs(s)
                    flrs.append(flr)

        return flrs

    def generate_flrg(self, flrs):
        for flr in flrs:
            flrg = mvflrg.FLRG(lhs=flr.LHS)

            if flrg.get_key() not in self.flrgs:
                self.flrgs[flrg.get_key()] = flrg

            self.flrgs[flrg.get_key()].append_rhs(flr.RHS)


    def train(self, data, **kwargs):

        ndata = self.apply_transformations(data)

        flrs = self.generate_flrs(ndata)
        self.generate_flrg(flrs)

    def forecast(self, data, **kwargs):
        ret = []
        ndata = self.apply_transformations(data)
        for ix in ndata.index:
            data_point = ndata.loc[ix]
            flrs = self.generate_lhs_flrs(data_point)
            mvs = []
            mps = []
            for flr in flrs:
                flrg = mvflrg.FLRG(lhs=flr.LHS)
                if flrg.get_key() not in self.flrgs:
                    #print('hit')
                    mvs.append(0.)
                    mps.append(0.)
                else:
                    mvs.append(self.flrgs[flrg.get_key()].get_membership(self.format_data(data_point), self.explanatory_variables))
                    mps.append(self.flrgs[flrg.get_key()].get_midpoint(self.target_variable.partitioner.sets))

            #print('mv', mvs)
            #print('mp', mps)
            mv = np.array(mvs)
            mp = np.array(mps)

            ret.append(np.dot(mv,mp.T)/np.sum(mv))

        ret = self.target_variable.apply_inverse_transformations(ret,
                                                           params=data[self.target_variable.data_label].values)
        return ret

    def clone_parameters(self, model):
        super(MVFTS, self).clone_parameters(model)

        self.explanatory_variables = model.explanatory_variables
        self.target_variable = model.target_variable

    def __str__(self):
        _str = self.name + ":\n"
        for k in self.flrgs.keys():
            _str += str(self.flrgs[k]) + "\n"

        return _str


