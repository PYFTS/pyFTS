#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
import math
from operator import itemgetter
from pyFTS.common import FLR, FuzzySet
from pyFTS.models import hofts, ifts
from pyFTS.probabilistic import ProbabilityDistribution
from itertools import product


class ProbabilisticWeightedFLRG(hofts.HighOrderFLRG):
    """High Order Probabilistic Weighted Fuzzy Logical Relationship Group"""
    def __init__(self, order):
        super(ProbabilisticWeightedFLRG, self).__init__(order)
        self.RHS = {}
        self.rhs_count = {}
        self.frequency_count = 0.0
        self.Z = None

    def get_membership(self, data, sets):
        if isinstance(data, (np.ndarray, list)):
            return np.nanprod([sets[key].membership(data[count])
                               for count, key in enumerate(self.LHS, start=0)])
        else:
            return sets[self.LHS[0]].membership(data)

    def append_rhs(self, c, **kwargs):
        mv = kwargs.get('mv', 1.0)
        self.frequency_count += mv
        if c in self.RHS:
            self.rhs_count[c] += mv
        else:
            self.RHS[c] = c
            self.rhs_count[c] = mv

    def lhs_conditional_probability(self, x, sets, norm, uod, nbins):
        pk = self.frequency_count / norm

        tmp = pk * (self.get_membership(x, sets) / self.partition_function(sets, uod, nbins=nbins))

        return tmp

    def rhs_unconditional_probability(self, c):
        return self.rhs_count[c] / self.frequency_count

    def rhs_conditional_probability(self, x, sets, uod, nbins):
        total = 0.0
        for rhs in self.RHS:
            set = sets[rhs]
            wi = self.rhs_unconditional_probability(rhs)
            mv = set.membership(x) / set.partition_function(uod, nbins=nbins)
            total += wi * mv

        return total

    def partition_function(self, sets, uod, nbins=100):
        if self.Z is None:
            self.Z = 0.0
            for k in np.linspace(uod[0], uod[1], nbins):
                for key in self.LHS:
                    self.Z += sets[key].membership(k)

        return self.Z

    def get_midpoint(self, sets):
        '''Return the expectation of the PWFLRG, the weighted sum'''
        if self.midpoint is None:
            self.midpoint = np.sum(np.array([self.rhs_unconditional_probability(s) * sets[s].centroid
                                             for s in self.RHS]))

        return self.midpoint

    def get_upper(self, sets):
        if self.upper is None:
            self.upper = np.sum(np.array([self.rhs_unconditional_probability(s) * sets[s].upper for s in self.RHS]))

        return self.upper

    def get_lower(self, sets):
        if self.lower is None:
            self.lower = np.sum(np.array([self.rhs_unconditional_probability(s) * sets[s].lower for s in self.RHS]))

        return self.lower

    def __str__(self):
        tmp2 = ""
        for c in sorted(self.RHS):
            if len(tmp2) > 0:
                tmp2 = tmp2 + ", "
            tmp2 = tmp2 + "(" + str(round(self.rhs_count[c] / self.frequency_count, 3)) + ")" + c
        return self.get_key() + " -> " + tmp2


class ProbabilisticWeightedFTS(ifts.IntervalFTS):
    """High Order Probabilistic Weighted Fuzzy Time Series"""
    def __init__(self, **kwargs):
        super(ProbabilisticWeightedFTS, self).__init__(**kwargs)
        self.shortname = "PWFTS"
        self.name = "Probabilistic FTS"
        self.detail = "Silva, P.; Guimarães, F.; Sadaei, H."
        self.flrgs = {}
        self.global_frequency_count = 0
        self.has_point_forecasting = True
        self.has_interval_forecasting = True
        self.has_probability_forecasting = True
        self.is_high_order = True
        self.min_order = 1
        self.auto_update = kwargs.get('update',False)
        self.configure_lags(**kwargs)

    def train(self, data, **kwargs):

        self.configure_lags(**kwargs)
        parameters = kwargs.get('parameters','fuzzy')

        if parameters == 'monotonic':
            tmpdata = self.partitioner.fuzzyfy(data, mode='sets', method='maximum')
            flrs = FLR.generate_recurrent_flrs(tmpdata)
            self.generate_flrg(flrs)
        else:
            self.generate_flrg(data)

    def generate_lhs_flrg(self, sample, explain=False):
        nsample = [self.partitioner.fuzzyfy(k, mode="sets", alpha_cut=self.alpha_cut)
                   for k in sample]

        return self.generate_lhs_flrg_fuzzyfied(nsample, explain)

    def generate_lhs_flrg_fuzzyfied(self, sample, explain=False):
        lags = []

        flrgs = []

        for ct, o in enumerate(self.lags):
            lhs = sample[o - 1]
            lags.append( lhs )

            if explain:
                print("\t (Lag {}) {} -> {} \n".format(o, sample[o-1], lhs))

        # Trace the possible paths
        for path in product(*lags):
            flrg = ProbabilisticWeightedFLRG(self.order)

            for lhs in path:
                flrg.append_lhs(lhs)

            flrgs.append(flrg)

        return flrgs

    def generate_flrg(self, data):
        l = len(data)
        for k in np.arange(self.max_lag, l):
            if self.dump: print("FLR: " + str(k))

            sample = data[k - self.max_lag: k]

            flrgs = self.generate_lhs_flrg(sample)

            for flrg in flrgs:

                lhs_mv = flrg.get_membership(sample, self.partitioner.sets)

                if flrg.get_key() not in self.flrgs:
                    self.flrgs[flrg.get_key()] = flrg;

                fuzzyfied = self.partitioner.fuzzyfy(data[k], mode='both', method='fuzzy',
                                                     alpha_cut=self.alpha_cut)

                mvs = []
                for set, mv in fuzzyfied:
                    self.flrgs[flrg.get_key()].append_rhs(set, mv=lhs_mv * mv)
                    mvs.append(mv)

                tmp_fq = sum([lhs_mv*kk for kk in mvs if kk > 0])

                self.global_frequency_count += tmp_fq


    def update_model(self,data):
        pass


    def add_new_PWFLGR(self, flrg):
        if flrg.get_key() not in self.flrgs:
            tmp = ProbabilisticWeightedFLRG(self.order)
            for fs in flrg.LHS: tmp.append_lhs(fs)
            tmp.append_rhs(flrg.LHS[-1])
            self.flrgs[tmp.get_key()] = tmp;
            self.global_frequency_count += 1

    def flrg_lhs_unconditional_probability(self, flrg):
        if flrg.get_key() in self.flrgs:
            return self.flrgs[flrg.get_key()].frequency_count / self.global_frequency_count
        else:
            return 0.0
            #self.add_new_PWFLGR(flrg)
            #return self.flrg_lhs_unconditional_probability(flrg)

    def flrg_lhs_conditional_probability(self, x, flrg):
        mv = flrg.get_membership(x, self.sets)
        pb = self.flrg_lhs_unconditional_probability(flrg)
        return mv * pb

    def get_midpoint(self, flrg):
        if flrg.get_key() in self.flrgs:
            tmp = self.flrgs[flrg.get_key()]
            ret = tmp.get_midpoint(self.sets) #sum(np.array([tmp.rhs_unconditional_probability(s) * self.setsDict[s].centroid for s in tmp.RHS]))
        else:
            if len(flrg.LHS) > 0:
                pi = 1 / len(flrg.LHS)
                ret = sum(np.array([pi * self.sets[s].centroid for s in flrg.LHS]))
            else:
                ret = np.nan
        return ret

    def flrg_rhs_conditional_probability(self, x, flrg):

        if flrg.get_key() in self.flrgs:
            _flrg = self.flrgs[flrg.get_key()]
            cond = []
            for s in _flrg.RHS.keys():
                _set = self.sets[s]
                tmp = _flrg.rhs_unconditional_probability(s) * (_set.membership(x) / _set.partition_function(uod=self.get_UoD()))
                cond.append(tmp)
            ret = sum(np.array(cond))
        else:
            pi = 1 / len(flrg.LHS)
            ret = sum(np.array([pi * self.sets[s].membership(x) for s in flrg.LHS]))
        return ret

    def get_upper(self, flrg):
        if flrg.get_key() in self.flrgs:
            tmp = self.flrgs[flrg.get_key()]
            ret = tmp.get_upper(self.sets)
        else:
            ret = 0
        return ret

    def get_lower(self, flrg):
        if flrg.get_key() in self.flrgs:
            tmp = self.flrgs[flrg.get_key()]
            ret = tmp.get_lower(self.sets)
        else:
            ret = 0
        return ret

    def forecast(self, data, **kwargs):
        method = kwargs.get('method','heuristic')

        l = len(data)

        ret = []

        for k in np.arange(self.max_lag - 1, l):
            sample = data[k - (self.max_lag - 1): k + 1]

            if method == 'heuristic':
                ret.append(self.point_heuristic(sample, **kwargs))
            elif method == 'expected_value':
                ret.append(self.point_expected_value(sample, **kwargs))
            else:
                raise ValueError("Unknown point forecasting method!")

            if self.auto_update and k > self.order+1: self.update_model(data[k - self.order - 1 : k])

        return ret

    def point_heuristic(self, sample, **kwargs):

        explain = kwargs.get('explain', False)

        if explain:
            print("Fuzzyfication \n")

        flrgs = self.generate_lhs_flrg(sample, explain)

        mp = []
        norms = []

        if explain:
            print("Rules:\n")

        for flrg in flrgs:
            norm = self.flrg_lhs_conditional_probability(sample, flrg)

            if norm == 0:
                norm = self.flrg_lhs_unconditional_probability(flrg)


            if explain:
                print("\t {} \t Midpoint: {}\t Norm: {}\n".format(str(self.flrgs[flrg.get_key()]),
                                                                  self.get_midpoint(flrg), norm))

            mp.append(norm * self.get_midpoint(flrg))
            norms.append(norm)

        norm = sum(norms)

        final = sum(mp) / norm if norm != 0 else 0

        if explain:
            print("Deffuzyfied value: {} \n".format(final))
        return final

    def point_expected_value(self, sample, **kwargs):
        explain = kwargs.get('explain', False)

        dist = self.forecast_distribution(sample)[0]

        final = dist.expected_value()
        return final

    def forecast_interval(self, ndata, **kwargs):

        method = kwargs.get('method','heuristic')
        alpha = kwargs.get('alpha', 0.05)

        l = len(ndata)

        ret = []

        for k in np.arange(self.max_lag - 1, l):

            sample = ndata[k - (self.max_lag - 1): k + 1]

            if method == 'heuristic':
                ret.append(self.interval_heuristic(sample))
            elif method == 'quantile':
                ret.append(self.interval_quantile(sample, alpha))
            else:
                raise ValueError("Unknown interval forecasting method!")

        return ret

    def interval_quantile(self, ndata, alpha):
        dist = self.forecast_distribution(ndata)
        itvl = dist[0].quantile([alpha, 1.0 - alpha])
        return itvl

    def interval_heuristic(self, sample):

        flrgs = self.generate_lhs_flrg(sample)

        up = []
        lo = []
        norms = []
        for flrg in flrgs:
            norm = self.flrg_lhs_conditional_probability(sample, flrg)
            if norm == 0:
                norm = self.flrg_lhs_unconditional_probability(flrg)
            up.append(norm * self.get_upper(flrg))
            lo.append(norm * self.get_lower(flrg))
            norms.append(norm)

            # gerar o intervalo
        norm = sum(norms)
        if norm == 0:
            return [0, 0]
        else:
            lo_ = sum(lo) / norm
            up_ = sum(up) / norm
            return [lo_, up_]

    def forecast_distribution(self, ndata, **kwargs):

        smooth = kwargs.get("smooth", "none")

        l = len(ndata)
        uod = self.get_UoD()

        if 'bins' in kwargs:
            _bins = kwargs.pop('bins')
            nbins = len(_bins)
        else:
            nbins = kwargs.get("num_bins", 100)
            _bins = np.linspace(uod[0], uod[1], nbins)

        ret = []

        for k in np.arange(self.max_lag - 1, l):
            sample = ndata[k - (self.max_lag - 1): k + 1]

            flrgs = self.generate_lhs_flrg(sample)

            if 'type' in kwargs:
                kwargs.pop('type')

            dist = ProbabilityDistribution.ProbabilityDistribution(smooth, uod=uod, bins=_bins, **kwargs)

            for bin in _bins:
                num = []
                den = []
                for s in flrgs:
                    if s.get_key() in self.flrgs:
                        flrg = self.flrgs[s.get_key()]
                        pk = flrg.lhs_conditional_probability(sample, self.sets, self.global_frequency_count, uod, nbins)
                        wi = flrg.rhs_conditional_probability(bin, self.sets, uod, nbins)
                        num.append(wi * pk)
                        den.append(pk)
                    else:
                        num.append(0.0)
                        den.append(0.000000001)
                pf = sum(num) / sum(den)

                dist.set(bin, pf)

            ret.append(dist)

        return ret

    def __check_point_bounds(self, point):
        lower_set = self.partitioner.lower_set()
        upper_set = self.partitioner.upper_set()
        return point <= lower_set.lower or point >= upper_set.upper

    def forecast_ahead(self, data, steps, **kwargs):

        l = len(data)

        start = kwargs.get('start', self.max_lag)

        ret = data[start - self.max_lag: start].tolist()

        for k in np.arange(self.max_lag, steps+self.max_lag):

            if self.__check_point_bounds(ret[-1]) :
                ret.append(ret[-1])
            else:
                mp = self.forecast(ret[k - self.max_lag: k], **kwargs)
                ret.append(mp[0])

        return ret[self.max_lag:]

    def __check_interval_bounds(self, interval):
        if len(self.transformations) > 0:
            lower_set = self.partitioner.lower_set()
            upper_set = self.partitioner.upper_set()
            return interval[0] <= lower_set.lower and interval[1] >= upper_set.upper
        elif len(self.transformations) == 0:
            return interval[0] <= self.original_min and interval[1] >= self.original_max

    def forecast_ahead_interval(self, data, steps, **kwargs):

        l = len(data)

        start = kwargs.get('start', self.max_lag)

        sample = data[start - self.max_lag: start]

        ret = [[k, k] for k in sample]
        
        ret.append(self.forecast_interval(sample)[0])

        for k in np.arange(self.max_lag+1, steps+self.max_lag):

            if len(ret) > 0 and self.__check_interval_bounds(ret[-1]):
                ret.append(ret[-1])
            else:
                lower = self.forecast_interval([ret[x][0] for x in np.arange(k - self.max_lag, k)], **kwargs)
                upper = self.forecast_interval([ret[x][1] for x in np.arange(k - self.max_lag, k)], **kwargs)

                ret.append([np.min(lower), np.max(upper)])

        return ret[self.order:]

    def forecast_ahead_distribution(self, ndata, steps, **kwargs):

        ret = []

        smooth = kwargs.get("smooth", "none")

        uod = self.get_UoD()

        if 'bins' in kwargs:
            _bins = kwargs.pop('bins')
            nbins = len(_bins)
        else:
            nbins = kwargs.get("num_bins", 100)
            _bins = np.linspace(uod[0], uod[1], nbins)

        start = kwargs.get('start', self.max_lag)

        sample = ndata[start - self.max_lag: start]

        for dat in sample:
            if 'type' in kwargs:
                kwargs.pop('type')
            tmp = ProbabilityDistribution.ProbabilityDistribution(smooth, uod=uod, bins=_bins, **kwargs)
            tmp.set(dat, 1.0)
            ret.append(tmp)

        dist = self.forecast_distribution(sample, bins=_bins)[0]

        ret.append(dist)

        for k in np.arange(self.max_lag+1, steps+self.max_lag+1):
            dist = ProbabilityDistribution.ProbabilityDistribution(smooth, uod=uod, bins=_bins, **kwargs)

            lags = []

            # Find all bins of past distributions with probability greater than zero

            for ct, lag in enumerate(self.lags):
                dd = ret[k - lag]
                vals = [float(v) for v in dd.bins if np.round(dd.density(v), 4) > 0.0]
                lags.append( sorted(vals) )


            # Trace all possible combinations between the bins of past distributions

            for path in product(*lags):

                # get the combined probabilities for this path
                pk = np.prod([ret[k - (self.max_lag + lag)].density(path[ct])
                              for ct, lag in enumerate(self.lags)])


                d = self.forecast_distribution(path)[0]

                for bin in _bins:
                    dist.set(bin, dist.density(bin) + pk * d.density(bin))

            ret.append(dist)

        return ret[self.order:]

    def __str__(self):
        tmp = self.name + ":\n"
        for r in sorted(self.flrgs):
            p = round(self.flrgs[r].frequency_count / self.global_frequency_count, 3)
            tmp = tmp + "(" + str(p) + ") " + str(self.flrgs[r]) + "\n"
        return tmp


def visualize_distributions(model, **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import seaborn as sns

    ordered_sets = model.partitioner.ordered_sets
    ftpg_keys = sorted(model.flrgs.keys(), key=lambda x: model.flrgs[x].get_midpoint(model.sets))

    lhs_probs = [model.flrg_lhs_unconditional_probability(model.flrgs[k])
                 for k in ftpg_keys]

    mat = np.zeros((len(ftpg_keys), len(ordered_sets)))
    for row, w in enumerate(ftpg_keys):
        for col, k in enumerate(ordered_sets):
            if k in model.flrgs[w].RHS:
                mat[row, col] = model.flrgs[w].rhs_unconditional_probability(k)

    size = kwargs.get('size', (5,10))

    fig = plt.figure(figsize=size)

    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 4])
    ax1 = plt.subplot(gs[0])
    sns.barplot(x='y', y='x', color='darkblue', data={'x': ftpg_keys, 'y': lhs_probs}, ax=ax1)
    ax1.set_ylabel("LHS Probabilities")

    ind_sets = range(len(ordered_sets))
    ax = plt.subplot(gs[1])
    sns.heatmap(mat, cmap='Blues', ax=ax, yticklabels=False)
    ax.set_title("RHS probabilities")
    ax.set_xticks(ind_sets)
    ax.set_xticklabels(ordered_sets)
    ax.grid(True)
    ax.xaxis.set_tick_params(rotation=90)
