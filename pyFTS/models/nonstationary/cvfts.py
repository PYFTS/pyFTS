import numpy as np
from pyFTS.models import chen
from pyFTS.models.nonstationary import common,nsfts
from pyFTS.common import FLR


class ConditionalVarianceFTS(chen.ConventionalFTS):
    def __init__(self, name, **kwargs):
        super(ConditionalVarianceFTS, self).__init__("CVFTS " + name, **kwargs)
        self.name = "Conditional Variance FTS"
        self.detail = ""
        self.flrgs = {}
        #self.append_transformation(Transformations.Differential(1))
        if self.partitioner is None:
            self.min_tx = None
            self.max_tx = None
        else:
            self.min_tx = self.partitioner.min
            self.max_tx = self.partitioner.max
            self.append_transformation(self.partitioner.transformation)

        self.min_stack = [0,0,0]
        self.max_stack = [0,0,0]

    def train(self, data, **kwargs):
        if kwargs.get('sets', None) is not None:
            self.sets = kwargs.get('sets', None)

        ndata = self.apply_transformations(data)

        self.min_tx = min(ndata)
        self.max_tx = max(ndata)

        tmpdata = common.fuzzySeries(ndata, self.sets, method='fuzzy', const_t=0)
        flrs = FLR.generate_non_recurrent_flrs(tmpdata)
        self.generate_flrg(flrs)

    def generate_flrg(self, flrs, **kwargs):
        for flr in flrs:
            if flr.LHS.name in self.flrgs:
                self.flrgs[flr.LHS.name].append_rhs(flr.RHS)
            else:
                self.flrgs[flr.LHS.name] = nsfts.ConventionalNonStationaryFLRG(flr.LHS)
                self.flrgs[flr.LHS.name].append_rhs(flr.RHS)

    def _smooth(self, a):
        return .1 * a[0] + .3 * a[1] + .6 * a[2]

    def perturbation_factors(self, data):
        _max = 0
        _min = 0
        if data < self.min_tx:
            _min = data - self.min_tx if data < 0 else self.min_tx - data
        elif data > self.max_tx:
            _max = data - self.max_tx if data > 0 else self.max_tx - data
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

        affected_sets = [[ct, set.membership(sample, perturb[ct])]
                         for ct, set in enumerate(self.sets)
                         if set.membership(sample, perturb[ct]) > 0.0]

        if len(affected_sets) == 0:
            if sample < self.sets[0].get_lower(perturb[0]):
                affected_sets.append([0, 1])
            elif sample < self.sets[-1].get_lower(perturb[-1]):
                affected_sets.append([len(self.sets) - 1, 1])


        return affected_sets

    def forecast(self, data, **kwargs):
        ndata = np.array(self.apply_transformations(data))

        l = len(ndata)

        ret = []

        for k in np.arange(0, l):

            sample = ndata[k]

            perturb = self.perturbation_factors(sample)

            affected_sets = self._affected_sets(sample, perturb)

            tmp = []

            if len(affected_sets) == 1:
                ix = affected_sets[0][0]
                aset = self.sets[ix]
                if aset.name in self.flrgs:
                    tmp.append(self.flrgs[aset.name].get_midpoint(perturb[ix]))
                else:
                    print('naive')
                    tmp.append(aset.get_midpoint(perturb[ix]))
            else:
                for aset in affected_sets:
                    ix = aset[0]
                    fs = self.sets[ix]
                    tdisp = perturb[ix]
                    if fs.name in self.flrgs:
                        tmp.append(self.flrgs[fs.name].get_midpoint(tdisp) * aset[1])
                    else:
                        tmp.append(fs.get_midpoint(tdisp) * aset[1])

            pto = sum(tmp)

            ret.append(pto)

        ret = self.apply_inverse_transformations(ret, params=[data[self.order - 1:]])

        return ret


    def forecast_interval(self, data, **kwargs):
        ndata = np.array(self.apply_transformations(data))

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
                aset = self.sets[ix]
                if aset.name in self.flrgs:
                    lower.append(self.flrgs[aset.name].get_lower(perturb[ix]))
                    upper.append(self.flrgs[aset.name].get_upper(perturb[ix]))
                else:
                    lower.append(aset.get_lower(perturb[ix]))
                    upper.append(aset.get_upper(perturb[ix]))
            else:
                for aset in affected_sets:
                    ix = aset[0]
                    fs = self.sets[ix]
                    tdisp = perturb[ix]
                    if fs.name in self.flrgs:
                        lower.append(self.flrgs[fs.name].get_lower(tdisp) * aset[1])
                        upper.append(self.flrgs[fs.name].get_upper(tdisp) * aset[1])
                    else:
                        lower.append(fs.get_lower(tdisp) * aset[1])
                        upper.append(fs.get_upper(tdisp) * aset[1])

            itvl = [sum(lower), sum(upper)]

            ret.append(itvl)

        ret = self.apply_inverse_transformations(ret, params=[data[self.order - 1:]])

        return ret
