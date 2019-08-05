from pyFTS.common import fts, FuzzySet, FLR, Membership
from pyFTS.partitioners import Grid
from pyFTS.models.multivariate import FLR as MVFLR, common, flrg as mvflrg
from itertools import product
from types import LambdaType
from copy import deepcopy

import numpy as np
import pandas as pd


def product_dict(**kwargs):
    """
    Code by Seth Johnson
    :param kwargs:
    :return:
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))


class MVFTS(fts.FTS):
    """
    Multivariate extension of Chen's ConventionalFTS method
    """
    def __init__(self, **kwargs):
        super(MVFTS, self).__init__(**kwargs)
        self.explanatory_variables = kwargs.get('explanatory_variables',[])
        self.target_variable = kwargs.get('target_variable',None)
        self.flrgs = {}
        self.is_multivariate = True
        self.shortname = "MVFTS"
        self.name = "Multivariate FTS"
        self.uod_clip = False

    def append_variable(self, var):
        """
        Append a new endogenous variable to the model

        :param var: variable object
        :return:
        """
        self.explanatory_variables.append(var)

    def format_data(self, data):
        ndata = {}
        for var in self.explanatory_variables:
            ndata[var.name] = var.partitioner.extractor(data[var.data_label])

        return ndata

    def apply_transformations(self, data, params=None, updateUoD=False, **kwargs):
        ndata = data.copy(deep=True)
        for var in self.explanatory_variables:
            try:
                values = ndata[var.data_label].values #if isinstance(ndata, pd.DataFrame) else ndata[var.data_label]
                if self.uod_clip and var.partitioner.type == 'common':
                    ndata[var.data_label] = np.clip(values,
                                                    var.partitioner.min, var.partitioner.max)

                ndata[var.data_label] = var.apply_transformations(values)
            except:
                pass

        return ndata

    def generate_lhs_flrs(self, data):
        flrs = []
        lags = {}
        for vc, var in enumerate(self.explanatory_variables):
            data_point = data[var.name]
            lags[var.name] = common.fuzzyfy_instance(data_point, var, tuples=False)

        for path in product_dict(**lags):
            flr = MVFLR.FLR()

            flr.LHS = path

            #for var, fset in path.items():
            #    flr.set_lhs(var, fset)

            if len(flr.LHS.keys()) == len(self.explanatory_variables):
                flrs.append(flr)

        return flrs

    def generate_flrs(self, data):
        flrs = []
        for ct in np.arange(1, len(data.index)):
            ix = data.index[ct-1]
            data_point = self.format_data( data.loc[ix] )

            tmp_flrs = self.generate_lhs_flrs(data_point)

            target_ix = data.index[ct]
            target_point = data[self.target_variable.data_label][target_ix]
            target = common.fuzzyfy_instance(target_point, self.target_variable)

            for flr in tmp_flrs:
                for v, s in target:
                    new_flr = deepcopy(flr)
                    new_flr.set_rhs(s)
                    flrs.append(new_flr)

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
        c = 0
        for index, row in ndata.iterrows() if isinstance(ndata, pd.DataFrame) else enumerate(ndata):
            data_point = self.format_data(row)
            flrs = self.generate_lhs_flrs(data_point)
            mvs = []
            mps = []
            for flr in flrs:
                flrg = mvflrg.FLRG(lhs=flr.LHS)
                if flrg.get_key() not in self.flrgs:
                    #Naïve approach is applied when no rules were found
                    if self.target_variable.name in flrg.LHS:
                        fs = flrg.LHS[self.target_variable.name]
                        fset = self.target_variable.partitioner.sets[fs]
                        mp = fset.centroid
                        mv = fset.membership(data_point[self.target_variable.name])
                        mvs.append(mv)
                        mps.append(mp)
                    else:
                        mvs.append(0.)
                        mps.append(0.)
                else:
                    _flrg = self.flrgs[flrg.get_key()]
                    mvs.append(_flrg.get_membership(data_point, self.explanatory_variables))
                    mps.append(_flrg.get_midpoint(self.target_variable.partitioner.sets))

            mv = np.array(mvs)
            mp = np.array(mps)

            ret.append(np.dot(mv,mp.T)/np.nansum(mv))

        ret = self.target_variable.apply_inverse_transformations(ret,
                                                           params=data[self.target_variable.data_label].values)
        return ret

    def forecast_ahead(self, data, steps, **kwargs):
        generators = kwargs.get('generators',None)

        if generators is None:
            raise Exception('You must provide parameter \'generators\'! generators is a dict where the keys' +
                            ' are the dataframe column names (except the target_variable) and the values are ' +
                            'lambda functions that accept one value (the actual value of the variable) '
                            ' and return the next value or trained FTS models that accept the actual values and '
                            'forecast new ones.')

        ndata = self.apply_transformations(data)

        start = kwargs.get('start_at', 0)

        ndata = ndata.iloc[start: start + self.max_lag]
        ret = []
        for k in np.arange(0, steps):
            sample = ndata.iloc[-self.max_lag:]
            tmp = self.forecast(sample, **kwargs)

            if isinstance(tmp, (list, np.ndarray)):
                tmp = tmp[-1]

            ret.append(tmp)

            new_data_point = {}

            for data_label in generators.keys():
                if data_label != self.target_variable.data_label:
                    if isinstance(generators[data_label], LambdaType):
                        last_data_point = ndata.loc[ndata.index[-1]]
                        new_data_point[data_label] = generators[data_label](last_data_point[data_label])
                    elif isinstance(generators[data_label], fts.FTS):
                        gen_model = generators[data_label]
                        last_data_point = sample.iloc[-gen_model.order:]

                        if not gen_model.is_multivariate:
                            last_data_point = last_data_point[data_label].values

                        new_data_point[data_label] = gen_model.forecast(last_data_point)[0]

            new_data_point[self.target_variable.data_label] = tmp

            ndata = ndata.append(new_data_point, ignore_index=True)

        return ret[-steps:]

    def forecast_interval(self, data, **kwargs):
        ret = []
        ndata = self.apply_transformations(data)
        c = 0
        for index, row in ndata.iterrows() if isinstance(ndata, pd.DataFrame) else enumerate(ndata):
            data_point = self.format_data(row)
            flrs = self.generate_lhs_flrs(data_point)
            mvs = []
            ups = []
            los = []
            for flr in flrs:
                flrg = mvflrg.FLRG(lhs=flr.LHS)
                if flrg.get_key() not in self.flrgs:
                    #Naïve approach is applied when no rules were found
                    if self.target_variable.name in flrg.LHS:
                        fs = flrg.LHS[self.target_variable.name]
                        fset = self.target_variable.partitioner.sets[fs]
                        up = fset.upper
                        lo = fset.lower
                        mv = fset.membership(data_point[self.target_variable.name])
                        mvs.append(mv)
                        ups.append(up)
                        los.append(lo)
                    else:
                        mvs.append(0.)
                        ups.append(0.)
                        los.append(0.)
                else:
                    _flrg = self.flrgs[flrg.get_key()]
                    mvs.append(_flrg.get_membership(data_point, self.explanatory_variables))
                    ups.append(_flrg.get_upper(self.target_variable.partitioner.sets))
                    los.append(_flrg.get_lower(self.target_variable.partitioner.sets))

            mv = np.array(mvs)
            up = np.dot(mv, np.array(ups).T) / np.nansum(mv)
            lo = np.dot(mv, np.array(los).T) / np.nansum(mv)

            ret.append([lo, up])

        ret = self.target_variable.apply_inverse_transformations(ret,
                                                           params=data[self.target_variable.data_label].values)
        return ret

    def forecast_ahead_interval(self, data, steps, **kwargs):
        generators = kwargs.get('generators', None)

        if generators is None:
            raise Exception('You must provide parameter \'generators\'! generators is a dict where the keys' +
                            ' are the dataframe column names (except the target_variable) and the values are ' +
                            'lambda functions that accept one value (the actual value of the variable) '
                            ' and return the next value or trained FTS models that accept the actual values and '
                            'forecast new ones.')

        ndata = self.apply_transformations(data)

        start = kwargs.get('start_at', 0)

        ret = []
        ix = ndata.index[start: start + self.max_lag]
        lo = ndata.loc[ix] #[ndata.loc[k] for k in ix]
        up = ndata.loc[ix] #[ndata.loc[k] for k in ix]
        for k in np.arange(0, steps):
            tmp_lo = self.forecast_interval(lo[-self.max_lag:], **kwargs)[0]
            tmp_up = self.forecast_interval(up[-self.max_lag:], **kwargs)[0]

            ret.append([min(tmp_lo), max(tmp_up)])

            new_data_point_lo = {}
            new_data_point_up = {}

            for data_label in generators.keys():
                if data_label != self.target_variable.data_label:
                    if isinstance(generators[data_label], LambdaType):
                        last_data_point_lo = lo.loc[lo.index[-1]]
                        new_data_point_lo[data_label] = generators[data_label](last_data_point_lo[data_label])
                        last_data_point_up = up.loc[up.index[-1]]
                        new_data_point_up[data_label] = generators[data_label](last_data_point_up[data_label])
                    elif isinstance(generators[data_label], fts.FTS):
                        model = generators[data_label]
                        last_data_point_lo = lo.loc[lo.index[-model.order:]]
                        last_data_point_up = up.loc[up.index[-model.order:]]

                        if not model.is_multivariate:
                            last_data_point_lo = last_data_point_lo[data_label].values
                            last_data_point_up = last_data_point_up[data_label].values

                        new_data_point_lo[data_label] = model.forecast(last_data_point_lo)[0]
                        new_data_point_up[data_label] = model.forecast(last_data_point_up)[0]

            new_data_point_lo[self.target_variable.data_label] = min(tmp_lo)
            new_data_point_up[self.target_variable.data_label] = max(tmp_up)

            lo = lo.append(new_data_point_lo, ignore_index=True)
            up = up.append(new_data_point_up, ignore_index=True)

        return ret[-steps:]

    def clone_parameters(self, model):
        super(MVFTS, self).clone_parameters(model)

        self.explanatory_variables = model.explanatory_variables
        self.target_variable = model.target_variable

    def __str__(self):
        _str = self.name + ":\n"
        for k in self.flrgs.keys():
            _str += str(self.flrgs[k]) + "\n"

        return _str


