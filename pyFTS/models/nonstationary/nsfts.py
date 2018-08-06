import numpy as np
from pyFTS.common import FLR, fts
from pyFTS.models.nonstationary import common, flrg


class ConventionalNonStationaryFLRG(flrg.NonStationaryFLRG):
    """First Order NonStationary Fuzzy Logical Relationship Group"""

    def __init__(self, LHS, **kwargs):
        super(ConventionalNonStationaryFLRG, self).__init__(1, **kwargs)
        self.LHS = LHS
        self.RHS = set()

    def get_key(self):
        return self.LHS

    def append_rhs(self, c, **kwargs):
        self.RHS.add(c)

    def __str__(self):
        tmp = self.LHS + " -> "
        tmp2 = ""
        for c in sorted(self.RHS):
            if len(tmp2) > 0:
                tmp2 = tmp2 + ","
            tmp2 = tmp2 + c
        return tmp + tmp2


class NonStationaryFTS(fts.FTS):
    """NonStationaryFTS Fuzzy Time Series"""
    def __init__(self, **kwargs):
        super(NonStationaryFTS, self).__init__(**kwargs)
        self.name = "Non Stationary FTS"
        self.shortname = "NSFTS"
        self.detail = ""
        self.flrgs = {}
        self.method = kwargs.get('method','conditional')
        self.is_high_order = False
        if self.partitioner is not None:
            self.append_transformation(self.partitioner.transformation)

        if self.method == 'conditional':
            self.min_stack = [0, 0, 0]
            self.max_stack = [0, 0, 0]
            self.uod_clip = False
            self.order = 1
            self.min_order = 1
            self.max_lag = 1
            self.inputs = []
            self.forecasts = []
            self.residuals = []
            self.variance_residual = 0.
            self.mean_residual = 0.
            self.memory_window = kwargs.get("memory_window", 5)

    def generate_flrg(self, flrs, **kwargs):
        for flr in flrs:
            if flr.LHS.name in self.flrgs:
                self.flrgs[flr.LHS.name].append_rhs(flr.RHS.name)
            else:
                self.flrgs[flr.LHS.name] = ConventionalNonStationaryFLRG(flr.LHS.name)
                self.flrgs[flr.LHS.name].append_rhs(flr.RHS.name)

    def _smooth(self, a):
        return .1 * a[0] + .3 * a[1] + .6 * a[2]

    def train(self, data, **kwargs):

        if self.method == 'unconditional':
            window_size = kwargs.get('parameters', 1)
            tmpdata = common.fuzzySeries(data, self.sets,
                                         self.partitioner.ordered_sets,
                                         window_size, method='fuzzy')
        else:
            tmpdata = common.fuzzySeries(data, self.sets,
                                 self.partitioner.ordered_sets,
                                 method='fuzzy', const_t=0)

        flrs = FLR.generate_non_recurrent_flrs(tmpdata)
        self.generate_flrg(flrs)

        if self.method == 'conditional':
            self.forecasts = self.forecast(data, no_update=True)
            self.residuals = np.array(data[1:]) - np.array(self.forecasts[:-1])

            self.variance_residual = np.var(self.residuals)  # np.max(self.residuals
            self.mean_residual = np.mean(self.residuals)

            self.residuals = self.residuals[-self.memory_window:].tolist()
            self.forecasts = self.forecasts[-self.memory_window:]
            self.inputs = np.array(data[-self.memory_window:]).tolist()

    def conditional_perturbation_factors(self, data, **kwargs):
        npart = len(self.partitioner.sets)
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

        translate = np.linspace(_min, _max, npart)

        var = np.std(self.residuals)

        var = 0 if var < 1 else var

        loc = (self.mean_residual + np.mean(self.residuals))

        location = [_range + w + loc + k for k in np.linspace(-var,var, npart) for w in translate]

        scale = [abs(location[0] - location[2])]
        scale.extend([abs(location[k - 1] - location[k + 1]) for k in np.arange(1, npart)])
        scale.append(abs(location[-1] - location[-3]))

        perturb = [[location[k], scale[k]] for k in np.arange(npart)]

        return perturb

    def _fsset_key(self, ix):
        return self.partitioner.ordered_sets[ix]

    def _affected_sets(self, sample, perturb):

        if self.method == 'conditional':

            affected_sets = [[ct, self.sets[self._fsset_key(ct)].membership(sample, perturb[ct])]
                             for ct in np.arange(len(self.partitioner.sets))
                             if self.sets[self._fsset_key(ct)].membership(sample, perturb[ct]) > 0.0]
            if len(affected_sets) == 0:

                if sample < self.partitioner.lower_set().get_lower(perturb[0]):
                    affected_sets.append([0, 1])
                elif sample > self.partitioner.upper_set().get_upper(perturb[-1]):
                    affected_sets.append([len(self.sets) - 1, 1])

        else:
            affected_sets = [[ct, self.sets[self._fsset_key(ct)].membership(sample, perturb)]
                             for ct in np.arange(len(self.partitioner.sets))
                             if self.sets[self._fsset_key(ct)].membership(sample, perturb) > 0.0]

            if len(affected_sets) == 0:

                if sample < self.partitioner.lower_set().get_lower(perturb):
                    affected_sets.append([0, 1])
                elif sample > self.partitioner.upper_set().get_upper(perturb):
                    affected_sets.append([len(self.sets) - 1, 1])

        return affected_sets

    def forecast(self, ndata, **kwargs):

        time_displacement = kwargs.get("time_displacement",0)

        window_size = kwargs.get("window_size", 1)

        no_update = kwargs.get("no_update", False)

        l = len(ndata)

        ret = []

        for k in np.arange(0, l):

            sample = ndata[k]

            if self.method == 'unconditional':
                perturb = common.window_index(k + time_displacement, window_size)
            elif self.method == 'conditional':
                if not no_update:
                    perturb = self.conditional_perturbation_factors(sample)
                else:
                    perturb = [[0, 1] for k in np.arange(len(self.partitioner.sets))]

            affected_sets = self._affected_sets(sample, perturb)

            numerator = []
            denominator = []

            if len(affected_sets) == 1:
                ix = affected_sets[0][0]
                aset = self.partitioner.ordered_sets[ix]
                if aset in self.flrgs:
                    numerator.append(self.flrgs[aset].get_midpoint(self.sets, perturb[ix]))
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
                        numerator.append(self.flrgs[fs].get_midpoint(self.sets, tdisp) * aset[1])
                    else:
                        fuzzy_set = self.sets[fs]
                        numerator.append(fuzzy_set.get_midpoint(tdisp) * aset[1])
                    denominator.append(aset[1])

            if sum(denominator) > 0:
                pto = sum(numerator) / sum(denominator)
            else:
                pto = sum(numerator)

            ret.append(pto)

            if self.method == 'conditional' and not no_update:
                self.forecasts.append(pto)
                self.residuals.append(self.inputs[-1] - self.forecasts[-1])
                self.inputs.append(sample)

                self.inputs.pop(0)
                self.forecasts.pop(0)
                self.residuals.pop(0)

        return ret

    def forecast_interval(self, ndata, **kwargs):

        time_displacement = kwargs.get("time_displacement", 0)

        window_size = kwargs.get("window_size", 1)

        l = len(ndata)

        ret = []

        for k in np.arange(0, l):

            # print("input: " + str(ndata[k]))

            tdisp = common.window_index(k + time_displacement, window_size)

            affected_sets = [[self.sets[key], self.sets[key].membership(ndata[k], tdisp)]
                             for key in self.partitioner.ordered_sets
                             if self.sets[key].membership(ndata[k], tdisp) > 0.0]

            if len(affected_sets) == 0:
                affected_sets.append([common.check_bounds(ndata[k], self.partitioner, tdisp), 1.0])

            upper = []
            lower = []

            if len(affected_sets) == 1:
                aset = affected_sets[0][0]
                if aset.name in self.flrgs:
                    lower.append(self.flrgs[aset.name].get_lower(tdisp))
                    upper.append(self.flrgs[aset.name].get_upper(tdisp))
                else:
                    lower.append(aset.get_lower(tdisp))
                    upper.append(aset.get_upper(tdisp))
            else:
                for aset in affected_sets:
                    if aset[0].name in self.flrgs:
                        lower.append(self.flrgs[aset[0].name].get_lower(tdisp) * aset[1])
                        upper.append(self.flrgs[aset[0].name].get_upper(tdisp) * aset[1])
                    else:
                        lower.append(aset[0].get_lower(tdisp) * aset[1])
                        upper.append(aset[0].get_upper(tdisp) * aset[1])


            ret.append([sum(lower), sum(upper)])

        return ret