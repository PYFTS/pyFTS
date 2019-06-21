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
        self.frequency_count = 0.0
        self.Z = None

    def get_membership(self, data, sets):
        if isinstance(data, (np.ndarray, list, tuple, set)):
            return np.nanprod([sets[key].membership(data[count])
                               for count, key in enumerate(self.LHS, start=0)])
        else:
            return sets[self.LHS[0]].membership(data)

    def append_rhs(self, c, **kwargs):
        count = kwargs.get('count', 1.0)
        self.frequency_count += count
        if c in self.RHS:
            self.RHS[c] += count
        else:
            self.RHS[c] = count

    def lhs_conditional_probability(self, x, sets, norm, uod, nbins):
        pk = self.frequency_count / norm

        tmp = pk * (self.get_membership(x, sets) / self.partition_function(sets, uod, nbins=nbins))

        return tmp

    def lhs_conditional_probability_fuzzyfied(self, lhs_mv, sets, norm, uod, nbins):
        pk = self.frequency_count / norm

        tmp = pk * (lhs_mv / self.partition_function(sets, uod, nbins=nbins))

        return tmp

    def rhs_unconditional_probability(self, c):
        return self.RHS[c] / self.frequency_count

    def rhs_conditional_probability(self, x, sets, uod, nbins):
        total = 0.0
        for rhs in self.RHS.keys():
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
                                             for s in self.RHS.keys()]))

        return self.midpoint

    def get_upper(self, sets):
        if self.upper is None:
            self.upper = np.sum(np.array([self.rhs_unconditional_probability(s) * sets[s].upper
                                          for s in self.RHS.keys()]))

        return self.upper

    def get_lower(self, sets):
        if self.lower is None:
            self.lower = np.sum(np.array([self.rhs_unconditional_probability(s) * sets[s].lower
                                          for s in self.RHS.keys()]))

        return self.lower

    def __str__(self):
        tmp2 = ""
        for c in sorted(self.RHS.keys()):
            if len(tmp2) > 0:
                tmp2 = tmp2 + ", "
            tmp2 = tmp2 + "(" + str(round(self.RHS[c] / self.frequency_count, 3)) + ")" + c
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

        if not kwargs.get('fuzzyfied',False):
            self.generate_flrg2(data)
        else:
            self.generate_flrg_fuzzyfied(data)

    def generate_flrg2(self, data):
        fuzz = []
        l = len(data)
        for k in np.arange(0, l):
            fuzz.append(self.partitioner.fuzzyfy(data[k], mode='both', method='fuzzy',
                                                 alpha_cut=self.alpha_cut))

        self.generate_flrg_fuzzyfied(fuzz)

    def generate_flrg_fuzzyfied(self, data):
        l = len(data)
        for k in np.arange(self.max_lag, l):
            sample = data[k - self.max_lag: k]
            set_sample = []
            for instance in sample:
                set_sample.append([k for k, v in instance])

            flrgs = self.generate_lhs_flrg_fuzzyfied(set_sample)

            for flrg in flrgs:

                if flrg.get_key() not in self.flrgs:
                    self.flrgs[flrg.get_key()] = flrg;

                lhs_mv = self.pwflrg_lhs_memberhip_fuzzyfied(flrg, sample)

                mvs = []
                inst = data[k]
                for set, mv in inst:
                    self.flrgs[flrg.get_key()].append_rhs(set, count=lhs_mv * mv)
                    mvs.append(mv)

                tmp_fq = sum([lhs_mv * kk for kk in mvs if kk > 0])

                self.global_frequency_count += tmp_fq

    def pwflrg_lhs_memberhip_fuzzyfied(self, flrg, sample):
        vals = []
        for ct, fuzz in enumerate(sample):
            vals.append([mv for fset, mv in fuzz if fset == flrg.LHS[ct]])

        return np.nanprod(vals)

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
                    self.flrgs[flrg.get_key()].append_rhs(set, count=lhs_mv * mv)
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
        mv = flrg.get_membership(x, self.partitioner.sets)
        pb = self.flrg_lhs_unconditional_probability(flrg)
        return mv * pb

    def flrg_lhs_conditional_probability_fuzzyfied(self, x, flrg):
        mv = self.pwflrg_lhs_memberhip_fuzzyfied(flrg, x)
        pb = self.flrg_lhs_unconditional_probability(flrg)
        return mv * pb

    def get_midpoint(self, flrg):
        if flrg.get_key() in self.flrgs:
            tmp = self.flrgs[flrg.get_key()]
            ret = tmp.get_midpoint(self.partitioner.sets) #sum(np.array([tmp.rhs_unconditional_probability(s) * self.setsDict[s].centroid for s in tmp.RHS]))
        else:
            if len(flrg.LHS) > 0:
                pi = 1 / len(flrg.LHS)
                ret = sum(np.array([pi * self.partitioner.sets[s].centroid for s in flrg.LHS]))
            else:
                ret = np.nan
        return ret

    def flrg_rhs_conditional_probability(self, x, flrg):

        if flrg.get_key() in self.flrgs:
            _flrg = self.flrgs[flrg.get_key()]
            cond = []
            for s in _flrg.RHS.keys():
                _set = self.partitioner.sets[s]
                tmp = _flrg.rhs_unconditional_probability(s) * (_set.membership(x) / _set.partition_function(uod=self.get_UoD()))
                cond.append(tmp)
            ret = sum(np.array(cond))
        else:
            pi = 1 / len(flrg.LHS)
            ret = sum(np.array([pi * self.partitioner.sets[s].membership(x) for s in flrg.LHS]))
        return ret

    def get_upper(self, flrg):
        if flrg.get_key() in self.flrgs:
            tmp = self.flrgs[flrg.get_key()]
            ret = tmp.get_upper(self.partitioner.sets)
        else:
            ret = 0
        return ret

    def get_lower(self, flrg):
        if flrg.get_key() in self.flrgs:
            tmp = self.flrgs[flrg.get_key()]
            ret = tmp.get_lower(self.partitioner.sets)
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
        fuzzyfied = kwargs.get('fuzzyfied', False)

        if explain:
            print("Fuzzyfication \n")

        if not fuzzyfied:
            flrgs = self.generate_lhs_flrg(sample, explain)
        else:
            fsets = self.get_sets_from_both_fuzzyfication(sample)
            flrgs = self.generate_lhs_flrg_fuzzyfied(fsets, explain)

        mp = []
        norms = []

        if explain:
            print("Rules:\n")

        for flrg in flrgs:
            if not fuzzyfied:
                norm = self.flrg_lhs_conditional_probability(sample, flrg)
            else:
                norm = self.flrg_lhs_conditional_probability_fuzzyfied(sample, flrg)

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

    def get_sets_from_both_fuzzyfication(self, sample):
        return [[k for k, v in inst] for inst in sample]

    def point_expected_value(self, sample, **kwargs):
        explain = kwargs.get('explain', False)

        dist = self.forecast_distribution(sample, **kwargs)[0]

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
                ret.append(self.interval_heuristic(sample, **kwargs))
            elif method == 'quantile':
                ret.append(self.interval_quantile(sample, alpha, **kwargs))
            else:
                raise ValueError("Unknown interval forecasting method!")

        return ret

    def interval_quantile(self, ndata, alpha, **kwargs):
        dist = self.forecast_distribution(ndata, **kwargs)
        itvl = dist[0].quantile([alpha, 1.0 - alpha])
        return itvl

    def interval_heuristic(self, sample, **kwargs):
        fuzzyfied = kwargs.get('fuzzyfied', False)

        if not fuzzyfied:
            flrgs = self.generate_lhs_flrg(sample)
        else:
            fsets = self.get_sets_from_both_fuzzyfication(sample)
            flrgs = self.generate_lhs_flrg_fuzzyfied(fsets)

        up = []
        lo = []
        norms = []
        for flrg in flrgs:
            if not fuzzyfied:
                norm = self.flrg_lhs_conditional_probability(sample, flrg)
            else:
                norm = self.flrg_lhs_conditional_probability_fuzzyfied(sample, flrg)

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

        fuzzyfied = kwargs.get('fuzzyfied', False)

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

            if not fuzzyfied:
                flrgs = self.generate_lhs_flrg(sample)
            else:
                fsets = self.get_sets_from_both_fuzzyfication(sample)
                flrgs = self.generate_lhs_flrg_fuzzyfied(fsets)

            if 'type' in kwargs:
                kwargs.pop('type')

            dist = ProbabilityDistribution.ProbabilityDistribution(smooth, uod=uod, bins=_bins, **kwargs)

            for bin in _bins:
                num = []
                den = []
                for s in flrgs:
                    if s.get_key() in self.flrgs:
                        flrg = self.flrgs[s.get_key()]
                        wi = flrg.rhs_conditional_probability(bin, self.partitioner.sets, uod, nbins)
                        if not fuzzyfied:
                            pk = flrg.lhs_conditional_probability(sample, self.partitioner.sets, self.global_frequency_count, uod, nbins)
                        else:
                            lhs_mv = self.pwflrg_lhs_memberhip_fuzzyfied(flrg, sample)
                            pk = flrg.lhs_conditional_probability_fuzzyfied(lhs_mv, self.partitioner.sets,
                                                                  self.global_frequency_count, uod, nbins)

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

        fuzzyfied = kwargs.get('fuzzyfied', False)

        start = kwargs.get('start_at', 0)

        ret = data[start: start+self.max_lag].tolist()

        for k in np.arange(self.max_lag, steps+self.max_lag):

            if self.__check_point_bounds(ret[-1]) and not fuzzyfied:
                ret.append(ret[-1])
            else:
                mp = self.forecast(ret[k - self.max_lag: k], **kwargs)
                ret.append(mp[0])

        return ret[-steps:]

    def __check_interval_bounds(self, interval):
        if len(self.transformations) > 0:
            lower_set = self.partitioner.lower_set()
            upper_set = self.partitioner.upper_set()
            return interval[0] <= lower_set.lower and interval[1] >= upper_set.upper
        elif len(self.transformations) == 0:
            return interval[0] <= self.original_min and interval[1] >= self.original_max

    def forecast_ahead_interval(self, data, steps, **kwargs):

        start = kwargs.get('start_at', 0)

        fuzzyfied = kwargs.get('fuzzyfied', False)

        sample = data[start: start + self.max_lag]

        if not fuzzyfied:
            ret = [[k, k] for k in sample]
        else:
            ret = []
            for k in sample:
                kv = self.partitioner.deffuzyfy(k,mode='both')
                ret.append([kv,kv])
        
        ret.append(self.forecast_interval(sample, **kwargs)[0])

        for k in np.arange(self.max_lag+1, steps+self.max_lag):

            if len(ret) > 0 and self.__check_interval_bounds(ret[-1]):
                ret.append(ret[-1])
            else:
                lower = self.forecast_interval([ret[x][0] for x in np.arange(k - self.max_lag, k)], **kwargs)
                upper = self.forecast_interval([ret[x][1] for x in np.arange(k - self.max_lag, k)], **kwargs)

                ret.append([np.min(lower), np.max(upper)])

        return ret[-steps:]

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

        start = kwargs.get('start_at', 0)

        sample = ndata[start: start + self.max_lag]

        for dat in sample:
            if 'type' in kwargs:
                kwargs.pop('type')
            tmp = ProbabilityDistribution.ProbabilityDistribution(smooth, uod=uod, bins=_bins, **kwargs)
            tmp.set(dat, 1.0)
            ret.append(tmp)

        dist = self.forecast_distribution(sample, bins=_bins, **kwargs)[0]

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

        return ret[-steps:]

    def __str__(self):
        tmp = self.name + ":\n"
        for r in sorted(self.flrgs.keys()):
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
