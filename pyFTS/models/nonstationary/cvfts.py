import numpy as np
from pyFTS.models import hofts
from pyFTS.models.nonstationary import common,nsfts
from pyFTS.common import FLR, flrg, tree

class HighOrderNonstationaryFLRG(hofts.HighOrderFTS):
    """Conventional High Order Fuzzy Logical Relationship Group"""
    def __init__(self, order, **kwargs):
        super(HighOrderNonstationaryFLRG, self).__init__(order, **kwargs)
        self.LHS = []
        self.RHS = {}
        self.strlhs = ""

    def append_rhs(self, c, **kwargs):
        if c not in self.RHS:
            self.RHS[c] = c

    def append_lhs(self, c):
        self.LHS.append(c)

    def __str__(self):
        tmp = ""
        for c in sorted(self.RHS):
            if len(tmp) > 0:
                tmp = tmp + ","
            tmp = tmp + c
        return self.get_key() + " -> " + tmp


    def __len__(self):
        return len(self.RHS)


class ConditionalVarianceFTS(hofts.HighOrderFTS):
    def __init__(self, **kwargs):
        super(ConditionalVarianceFTS, self).__init__(**kwargs)
        self.name = "Conditional Variance FTS"
        self.shortname = "CVFTS "
        self.detail = ""
        self.flrgs = {}
        self.is_high_order = False
        if self.partitioner is not None:
            self.append_transformation(self.partitioner.transformation)

        self.min_stack = [0,0,0]
        self.max_stack = [0,0,0]
        self.uod_clip = False
        self.order = 1
        self.min_order = 1
        self.max_lag = 1
        self.inputs = []
        self.forecasts = []
        self.residuals = []
        self.variance_residual = 0.
        self.mean_residual = 0.
        self.memory_window = kwargs.get("memory_window",5)

    def train(self, ndata, **kwargs):

        tmpdata = common.fuzzySeries(ndata, self.sets, self.partitioner.ordered_sets, method='fuzzy', const_t=0)
        flrs = FLR.generate_non_recurrent_flrs(tmpdata)
        self.generate_flrg(flrs)

        self.forecasts = self.forecast(ndata, no_update=True)
        self.residuals = np.array(ndata[1:]) - np.array(self.forecasts[:-1])

        self.variance_residual = np.var(self.residuals) # np.max(self.residuals
        self.mean_residual = np.mean(self.residuals)

        self.residuals = self.residuals[-self.memory_window:].tolist()
        self.forecasts = self.forecasts[-self.memory_window:]
        self.inputs = np.array(ndata[-self.memory_window:]).tolist()


    def generate_flrg(self, flrs, **kwargs):
        for flr in flrs:
            if flr.LHS.name in self.flrgs:
                self.flrgs[flr.LHS.name].append_rhs(flr.RHS)
            else:
                self.flrgs[flr.LHS.name] = nsfts.ConventionalNonStationaryFLRG(flr.LHS)
                self.flrgs[flr.LHS.name].append_rhs(flr.RHS)


    def _smooth(self, a):
        return .1 * a[0] + .3 * a[1] + .6 * a[2]

    def perturbation_factors(self, data, **kwargs):

        _max = 0
        _min = 0
        if data < self.original_min:
            _min = data - self.original_min if data < 0 else self.original_min - data
        elif data > self.original_max:
            _max = data - self.original_max if data > 0 else self.original_max - data
        self.min_stack.pop(2)
        self.min_stack.insert(0, _min)
        _min = min(self.min_stack)
        self.max_stack.pop(2)
        self.max_stack.insert(0, _max)
        _max = max(self.max_stack)

        _range = (_max - _min)/2

        translate = np.linspace(_min, _max, self.partitioner.partitions)

        var = np.std(self.residuals)

        var = 0 if var < 1 else var

        loc = (self.mean_residual + np.mean(self.residuals))

        location = [_range + w + loc + k for k in np.linspace(-var,var) for w in translate]

        perturb = [[location[k], var] for k in np.arange(0, self.partitioner.partitions)]

        return perturb

    def perturbation_factors__old(self, data):
        _max = 0
        _min = 0
        if data < self.original_min:
            _min = data - self.original_min if data < 0 else self.original_min - data
        elif data > self.original_max:
            _max = data - self.original_max if data > 0 else self.original_max - data
        self.min_stack.pop(2)
        self.min_stack.insert(0,_min)
        _min = min(self.min_stack)
        self.max_stack.pop(2)
        self.max_stack.insert(0, _max)
        _max = max(self.max_stack)

        location = np.linspace(_min, _max, self.partitioner.partitions)
        scale = [abs(location[0] - location[2])]
        scale.extend([abs(location[k-1] - location[k+1]) for k in np.arange(1,self.partitioner.partitions-1)])
        scale.append(abs(location[-1] - location[-3]))

        perturb = [[location[k], scale[k]] for k in np.arange(0, self.partitioner.partitions)]

        return perturb

    def _affected_sets(self, sample, perturb):

        affected_sets = [[ct, self.sets[key].membership(sample, perturb[ct])]
                         for ct, key in enumerate(self.partitioner.ordered_sets)
                         if self.sets[key].membership(sample, perturb[ct]) > 0.0]

        if len(affected_sets) == 0:
            if sample < self.partitioner.lower_set().get_lower(perturb[0]):
                affected_sets.append([0, 1])
            elif sample > self.partitioner.upper_set().get_upper(perturb[-1]):
                affected_sets.append([len(self.sets) - 1, 1])


        return affected_sets

    def forecast(self, ndata, **kwargs):
        l = len(ndata)

        ret = []

        no_update = kwargs.get("no_update",False)

        for k in np.arange(0, l):

            sample = ndata[k]

            if not no_update:
                perturb = self.perturbation_factors(sample)
            else:
                perturb = [[0, 1] for k in np.arange(0, self.partitioner.partitions)]

            affected_sets = self._affected_sets(sample, perturb)

            numerator = []
            denominator = []

            if len(affected_sets) == 1:
                ix = affected_sets[0][0]
                aset = self.partitioner.ordered_sets[ix]
                if aset in self.flrgs:
                    numerator.append(self.flrgs[aset].get_midpoint(perturb[ix]))
                else:
                    fuzzy_set = self.sets[aset]
                    numerator.append(fuzzy_set.get_midpoint(perturb[ix]))
                denominator.append(1)
            else:
                for aset in affected_sets:
                    ix = aset[0]
                    fs = self.partitioner.ordered_sets[ix]
                    tdisp = perturb[ix]
                    if fs in self.flrgs:
                        numerator.append(self.flrgs[fs].get_midpoint(tdisp) * aset[1])
                    else:
                        fuzzy_set = self.sets[fs]
                        numerator.append(fuzzy_set.get_midpoint(tdisp) * aset[1])
                    denominator.append(aset[1])

            if sum(denominator) > 0:
                pto = sum(numerator) /sum(denominator)
            else:
                pto = sum(numerator)

            ret.append(pto)

            if not no_update:
                self.forecasts.append(pto)
                self.residuals.append(self.inputs[-1] - self.forecasts[-1])
                self.inputs.append(sample)

                self.inputs.pop(0)
                self.forecasts.pop(0)
                self.residuals.pop(0)

        return ret


    def forecast_interval(self, ndata, **kwargs):
        l = len(ndata)

        ret = []

        for k in np.arange(0, l):

            sample = ndata[k]

            perturb = self.perturbation_factors(sample)

            affected_sets = self._affected_sets(sample, perturb)

            upper = []
            lower = []

            if len(affected_sets) == 1:
                ix = affected_sets[0][0]
                aset = self.partitioner.ordered_sets[ix]
                if aset in self.flrgs:
                    lower.append(self.flrgs[aset].get_lower(perturb[ix]))
                    upper.append(self.flrgs[aset].get_upper(perturb[ix]))
                else:
                    fuzzy_set = self.sets[aset]
                    lower.append(fuzzy_set.get_lower(perturb[ix]))
                    upper.append(fuzzy_set.get_upper(perturb[ix]))
            else:
                for aset in affected_sets:
                    ix = aset[0]
                    fs = self.partitioner.ordered_sets[ix]
                    tdisp = perturb[ix]
                    if fs in self.flrgs:
                        lower.append(self.flrgs[fs].get_lower(tdisp) * aset[1])
                        upper.append(self.flrgs[fs].get_upper(tdisp) * aset[1])
                    else:
                        fuzzy_set = self.sets[fs]
                        lower.append(fuzzy_set.get_lower(tdisp) * aset[1])
                        upper.append(fuzzy_set.get_upper(tdisp) * aset[1])

            itvl = [sum(lower), sum(upper)]

            ret.append(itvl)

        return ret