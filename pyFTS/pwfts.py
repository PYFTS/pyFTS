#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
import math
from operator import itemgetter
from pyFTS.common import FLR, FuzzySet, SortedCollection
from pyFTS import hofts, ifts, tree
from pyFTS.probabilistic import ProbabilityDistribution


class ProbabilisticWeightedFLRG(hofts.HighOrderFLRG):
    """High Order Probabilistic Weighted Fuzzy Logical Relationship Group"""
    def __init__(self, order):
        super(ProbabilisticWeightedFLRG, self).__init__(order)
        self.RHS = {}
        self.rhs_count = {}
        self.frequency_count = 0.0
        self.Z = None

    def appendRHS(self, c):
        self.frequency_count += 1.0
        if c.name in self.RHS:
            self.rhs_count[c.name] += 1.0
        else:
            self.RHS[c.name] = c
            self.rhs_count[c.name] = 1.0

    def appendRHSFuzzy(self, c, mv):
        self.frequency_count += mv
        if c.name in self.RHS:
            self.rhs_count[c.name] += mv
        else:
            self.RHS[c.name] = c
            self.rhs_count[c.name] = mv

    def get_RHSprobability(self, c):
        return self.rhs_count[c] / self.frequency_count

    def lhs_probability(self, x, norm, uod, nbins):
        pk = self.frequency_count / norm

        tmp = pk * (self.lhs_membership(x) / self.partition_function(uod, nbins=nbins))

        return tmp

    def rhs_conditional_probability(self, x, sets, uod, nbins):
        total = 0.0
        for rhs in self.RHS.keys():
            set = self.RHS[rhs]
            wi = self.get_RHSprobability(rhs)
            mv = set.membership(x) / set.partition_function(uod, nbins=nbins)
            total += wi * mv

        return total

    def lhs_membership(self,x):
        mv = []
        for count, set in enumerate(self.LHS):
            mv.append(set.membership(x[count]))

        min_mv = np.min(mv)
        return min_mv

    def partition_function(self, uod, nbins=100):
        if self.Z is None:
            self.Z = 0.0
            for k in np.linspace(uod[0], uod[1], nbins):
                for set in self.LHS:
                    self.Z += set.membership(k)

        return self.Z

    def get_midpoint(self):
        '''Return the expectation of the PWFLRG, the weighted sum'''
        return sum(np.array([self.get_RHSprobability(s) * self.RHS[s].centroid
                             for s in self.RHS.keys()]))

    def get_upper(self):
        return sum(np.array([self.get_RHSprobability(s) * self.RHS[s].upper
                             for s in self.RHS.keys()]))

    def get_lower(self):
        return sum(np.array([self.get_RHSprobability(s) * self.RHS[s].lower
                             for s in self.RHS.keys()]))

    def __str__(self):
        tmp2 = ""
        for c in sorted(self.RHS.keys()):
            if len(tmp2) > 0:
                tmp2 = tmp2 + ", "
            tmp2 = tmp2 + "(" + str(round(self.rhs_count[c] / self.frequency_count, 3)) + ")" + c
        return self.strLHS() + " -> " + tmp2


class ProbabilisticWeightedFTS(ifts.IntervalFTS):
    """High Order Probabilistic Weighted Fuzzy Time Series"""
    def __init__(self, name, **kwargs):
        super(ProbabilisticWeightedFTS, self).__init__(name=name, **kwargs)
        self.shortname = "PWFTS " + name
        self.name = "Probabilistic FTS"
        self.detail = "Silva, P.; Guimarães, F.; Sadaei, H."
        self.flrgs = {}
        self.global_frequency_count = 0
        self.has_point_forecasting = True
        self.has_interval_forecasting = True
        self.has_probability_forecasting = True
        self.is_high_order = True
        self.auto_update = kwargs.get('update',False)
        self.interval_method = kwargs.get('interval_method','extremum')
        self.alpha = kwargs.get('alpha', 0.05)

    def train(self, data, sets, order=1,parameters='Fuzzy'):

        data = self.doTransformations(data, updateUoD=True)

        self.order = order
        if sets is None and self.partitioner is not None:
            self.sets = self.partitioner.sets
            self.original_min = self.partitioner.min
            self.original_max = self.partitioner.max
        else:
            self.sets = sets
        for s in self.sets:    self.setsDict[s.name] = s
        if parameters == 'Monotonic':
            tmpdata = FuzzySet.fuzzySeries(data, sets)
            flrs = FLR.generateRecurrentFLRs(tmpdata)
            self.flrgs = self.generateFLRG(flrs)
        else:
            self.flrgs = self.generate_flrg(data)

    def generate_flrg(self, data):
        flrgs = {}
        l = len(data)
        for k in np.arange(self.order, l):
            if self.dump: print("FLR: " + str(k))

            sample = data[k - self.order: k]

            mvs = FuzzySet.fuzzyInstances(sample, self.sets)
            lags = {}

            mv = FuzzySet.fuzzyInstance(data[k], self.sets)
            tmp = np.argwhere(mv)
            idx = np.ravel(tmp)  # flatten the array

            for o in np.arange(0, self.order):
                _sets = [self.sets[kk] for kk in np.arange(0, len(self.sets)) if mvs[o][kk] > 0]

                lags[o] = _sets

            root = tree.FLRGTreeNode(None)

            self.build_tree_without_order(root, lags, 0)

            # Trace the possible paths
            for p in root.paths():
                flrg = ProbabilisticWeightedFLRG(self.order)
                path = list(reversed(list(filter(None.__ne__, p))))

                tmp_path = []
                for c, e in enumerate(path, start=0):
                    tmp_path.append(  e.membership( sample[c] )  )
                    flrg.appendLHS(e)

                lhs_mv = np.prod(tmp_path)

                if flrg.strLHS() not in flrgs:
                    flrgs[flrg.strLHS()] = flrg;

                for st in idx:
                    flrgs[flrg.strLHS()].appendRHSFuzzy(self.sets[st], lhs_mv*mv[st])

                tmp_fq = sum([lhs_mv*kk for kk in mv if kk > 0])

                self.global_frequency_count = self.global_frequency_count + tmp_fq

        return flrgs

    def generateFLRG(self, flrs):
        flrgs = {}
        l = len(flrs)
        for k in np.arange(self.order, l+1):
            if self.dump: print("FLR: " + str(k))
            flrg = ProbabilisticWeightedFLRG(self.order)

            for kk in np.arange(k - self.order, k):
                flrg.appendLHS(flrs[kk].LHS)
                if self.dump: print("LHS: " + str(flrs[kk]))

            if flrg.strLHS() in flrgs:
                flrgs[flrg.strLHS()].appendRHS(flrs[k-1].RHS)
            else:
                flrgs[flrg.strLHS()] = flrg
                flrgs[flrg.strLHS()].appendRHS(flrs[k-1].RHS)
            if self.dump: print("RHS: " + str(flrs[k-1]))

            self.global_frequency_count += 1
        return (flrgs)

    def update_model(self,data):

        fzzy = FuzzySet.fuzzySeries(data, self.sets)

        flrg = ProbabilisticWeightedFLRG(self.order)

        for k in np.arange(0, self.order): flrg.appendLHS(fzzy[k])

        if flrg.strLHS() in self.flrgs:
            self.flrgs[flrg.strLHS()].appendRHS(fzzy[self.order])
        else:
            self.flrgs[flrg.strLHS()] = flrg
            self.flrgs[flrg.strLHS()].appendRHS(fzzy[self.order])

        self.global_frequency_count += 1

    def add_new_PWFLGR(self, flrg):
        if flrg.strLHS() not in self.flrgs:
            tmp = ProbabilisticWeightedFLRG(self.order)
            for fs in flrg.LHS: tmp.appendLHS(fs)
            tmp.appendRHS(flrg.LHS[-1])
            self.flrgs[tmp.strLHS()] = tmp;
            self.global_frequency_count += 1

    def get_flrg_global_probability(self, flrg):
        if flrg.strLHS() in self.flrgs:
            return self.flrgs[flrg.strLHS()].frequency_count / self.global_frequency_count
        else:
            self.add_new_PWFLGR(flrg)
            return self.get_flrg_global_probability(flrg)

    def get_midpoint(self, flrg):
        if flrg.strLHS() in self.flrgs:
            tmp = self.flrgs[flrg.strLHS()]
            ret = tmp.get_midpoint() #sum(np.array([tmp.get_RHSprobability(s) * self.setsDict[s].centroid for s in tmp.RHS]))
        else:
            pi = 1 / len(flrg.LHS)
            ret = sum(np.array([pi * s.centroid for s in flrg.LHS]))
        return ret

    def get_conditional_probability(self, x, flrg):

        if flrg.strLHS() in self.flrgs:
            _flrg = self.flrgs[flrg.strLHS()]
            cond = []
            for s in _flrg.RHS:
                _set = self.setsDict[s]
                tmp = _flrg.get_RHSprobability(s) * (_set.membership(x) / _set.partition_function(uod=self.get_UoD()))
                cond.append(tmp)
            ret = sum(np.array(cond))
        else:
            ##########################################
            # this may be the problem! TEST IT!!!
            ##########################################
            pi = 1 / len(flrg.LHS)
            ret = sum(np.array([pi * self.setsDict[s].membership(x) for s in flrg.LHS]))
        return ret

    def get_upper(self, flrg):
        if flrg.strLHS() in self.flrgs:
            tmp = self.flrgs[flrg.strLHS()]
            ret = tmp.get_upper()
        else:
            pi = 1 / len(flrg.LHS)
            ret = sum(np.array([pi * s.upper for s in flrg.LHS]))
        return ret

    def get_lower(self, flrg):
        if flrg.strLHS() in self.flrgs:
            tmp = self.flrgs[flrg.strLHS()]
            ret = tmp.get_lower()
        else:
            pi = 1 / len(flrg.LHS)
            ret = sum(np.array([pi * s.lower for s in flrg.LHS]))
        return ret

    def forecast(self, data, **kwargs):

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)

        ret = []

        for k in np.arange(self.order - 1, l):

            # print(k)

            affected_flrgs = []
            affected_flrgs_memberships = []
            norms = []

            mp = []

            # Find the sets which membership > 0 for each lag
            lags = {}
            if self.order > 1:
                subset = ndata[k - (self.order - 1): k + 1]

                for count, instance in enumerate(subset):
                    mb = FuzzySet.fuzzyInstance(instance, self.sets)
                    tmp = np.argwhere(mb)
                    idx = np.ravel(tmp)  # flatten the array

                    if idx.size == 0:  # the element is out of the bounds of the Universe of Discourse
                        if instance <= self.sets[0].lower:
                            idx = [0]
                        elif instance >= self.sets[-1].upper:
                            idx = [len(self.sets) - 1]
                        else:
                            raise Exception(instance)

                    lags[count] = idx

                # Build the tree with all possible paths

                root = tree.FLRGTreeNode(None)

                self.build_tree(root, lags, 0)

                # Trace the possible paths and build the PFLRG's

                for p in root.paths():
                    path = list(reversed(list(filter(None.__ne__, p))))
                    flrg = hofts.HighOrderFLRG(self.order)
                    for kk in path: flrg.appendLHS(self.sets[kk])

                    assert len(flrg.LHS) == subset.size, str(subset) + " -> " + str([s.name for s in flrg.LHS])

                    ##
                    affected_flrgs.append(flrg)

                    # Find the general membership of FLRG
                    affected_flrgs_memberships.append(flrg.get_membership())

            else:

                mv = FuzzySet.fuzzyInstance(ndata[k], self.sets)  # get all membership values
                tmp = np.argwhere(mv)  # get the indices of values > 0
                idx = np.ravel(tmp)  # flatten the array

                if idx.size == 0:  # the element is out of the bounds of the Universe of Discourse
                    if ndata[k] <= self.sets[0].lower:
                        idx = [0]
                    elif ndata[k] >= self.sets[-1].upper:
                        idx = [len(self.sets) - 1]
                    else:
                        raise Exception(ndata[k])

                for kk in idx:
                    flrg = hofts.HighOrderFLRG(self.order)
                    flrg.appendLHS(self.sets[kk])
                    affected_flrgs.append(flrg)
                    affected_flrgs_memberships.append(mv[kk])

            for count, flrg in enumerate(affected_flrgs):
                # achar o os bounds de cada FLRG, ponderados pela probabilidade e pertinência
                norm = self.get_flrg_global_probability(flrg) * affected_flrgs_memberships[count]
                if norm == 0:
                    norm = self.get_flrg_global_probability(flrg)  # * 0.001
                mp.append(norm * self.get_midpoint(flrg))
                norms.append(norm)

            # gerar o intervalo
            norm = sum(norms)
            if norm == 0:
                ret.append(0)
            else:
                ret.append(sum(mp) / norm)

        if self.auto_update and k > self.order+1: self.update_model(ndata[k - self.order - 1 : k])

        ret = self.doInverseTransformations(ret, params=[data[self.order - 1:]])

        return ret

    def forecastInterval(self, data, **kwargs):

        if 'method' in kwargs:
            self.interval_method = kwargs.get('method','quantile')

        if 'alpha' in kwargs:
            self.alpha = kwargs.get('alpha', 0.05)

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)

        ret = []

        for k in np.arange(self.order - 1, l):

            if self.interval_method == 'extremum':
                self.interval_extremum(k, ndata, ret)
            else:
                self.interval_quantile(k, ndata, ret)

        ret = self.doInverseTransformations(ret, params=[data[self.order - 1:]], interval=True)

        return ret

    def interval_quantile(self, k, ndata, ret):
        dist = self.forecastDistribution(ndata)
        lo_qt = dist[0].quantile(self.alpha)
        up_qt = dist[0].quantile(1.0 - self.alpha)
        ret.append([lo_qt, up_qt])

    def interval_extremum(self, k, ndata, ret):
        affected_flrgs = []
        affected_flrgs_memberships = []
        norms = []
        up = []
        lo = []
        # Find the sets which membership > 0 for each lag
        count = 0
        lags = {}
        if self.order > 1:
            subset = ndata[k - (self.order - 1): k + 1]

            for instance in subset:
                mb = FuzzySet.fuzzyInstance(instance, self.sets)
                tmp = np.argwhere(mb)
                idx = np.ravel(tmp)  # flatten the array

                if idx.size == 0:  # the element is out of the bounds of the Universe of Discourse
                    if math.isclose(instance, self.sets[0].lower) or instance < self.sets[0].lower:
                        idx = [0]
                    elif math.isclose(instance, self.sets[-1].upper) or instance > self.sets[-1].upper:
                        idx = [len(self.sets) - 1]
                    else:
                        raise Exception("Data exceed the known bounds [%s, %s] of universe of discourse: %s" %
                                        (self.sets[0].lower, self.sets[-1].upper, instance))

                lags[count] = idx
                count += 1

            # Build the tree with all possible paths

            root = tree.FLRGTreeNode(None)

            self.build_tree(root, lags, 0)

            # Trace the possible paths and build the PFLRG's

            for p in root.paths():
                path = list(reversed(list(filter(None.__ne__, p))))
                flrg = hofts.HighOrderFLRG(self.order)
                for kk in path: flrg.appendLHS(self.sets[kk])

                assert len(flrg.LHS) == subset.size, str(subset) + " -> " + str([s.name for s in flrg.LHS])

                ##
                affected_flrgs.append(flrg)

                # Find the general membership of FLRG
                affected_flrgs_memberships.append(min(self.get_sequence_membership(subset, flrg.LHS)))

        else:

            mv = FuzzySet.fuzzyInstance(ndata[k], self.sets)  # get all membership values
            tmp = np.argwhere(mv)  # get the indices of values > 0
            idx = np.ravel(tmp)  # flatten the array

            if idx.size == 0:  # the element is out of the bounds of the Universe of Discourse
                if math.isclose(ndata[k], self.sets[0].lower) or ndata[k] < self.sets[0].lower:
                    idx = [0]
                elif math.isclose(ndata[k], self.sets[-1].upper) or ndata[k] > self.sets[-1].upper:
                    idx = [len(self.sets) - 1]
                else:
                    raise Exception("Data exceed the known bounds [%s, %s] of universe of discourse: %s" %
                                    (self.sets[0].lower, self.sets[-1].upper, ndata[k]))

            for kk in idx:
                flrg = hofts.HighOrderFLRG(self.order)
                flrg.appendLHS(self.sets[kk])
                affected_flrgs.append(flrg)
                affected_flrgs_memberships.append(mv[kk])
        for count, flrg in enumerate(affected_flrgs):
            # achar o os bounds de cada FLRG, ponderados pela probabilidade e pertinência
            norm = self.get_flrg_global_probability(flrg) * affected_flrgs_memberships[count]
            if norm == 0:
                norm = self.get_flrg_global_probability(flrg)  # * 0.001
            up.append(norm * self.get_upper(flrg))
            lo.append(norm * self.get_lower(flrg))
            norms.append(norm)

        # gerar o intervalo
        norm = sum(norms)
        if norm == 0:
            ret.append([0, 0])
        else:
            lo_ = sum(lo) / norm
            up_ = sum(up) / norm
            ret.append([lo_, up_])

    def forecastDistribution(self, data, **kwargs):

        if not isinstance(data, (list, set, np.ndarray)):
            data = [data]

        smooth = kwargs.get("smooth", "none")
        nbins = kwargs.get("num_bins", 100)

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)

        ret = []
        uod = self.get_UoD()
        _keys = sorted(self.flrgs.keys())
        _bins = np.linspace(uod[0], uod[1], nbins)

        for k in np.arange(self.order - 1, l):
            sample = ndata[k - (self.order - 1): k + 1]
            dist = ProbabilityDistribution.ProbabilityDistribution(smooth, uod=uod, bins=_bins, **kwargs)

            for bin in _bins:
                num = []
                den = []
                for s in _keys:
                    flrg = self.flrgs[s]
                    pk = flrg.lhs_probability(sample, self.global_frequency_count, uod, nbins)
                    wi = flrg.rhs_conditional_probability(bin, self.setsDict, uod, nbins)
                    num.append(wi * pk)
                    den.append(pk)
                pf = sum(num) / sum(den)

                dist.set(bin, pf)

            ret.append(dist)

        return ret

    def forecastAhead(self, data, steps, **kwargs):
        ret = [data[k] for k in np.arange(len(data) - self.order, len(data))]

        for k in np.arange(self.order - 1, steps):

            if ret[-1] <= self.sets[0].lower or ret[-1] >= self.sets[-1].upper:
                ret.append(ret[-1])
            else:
                mp = self.forecast([ret[x] for x in np.arange(k - self.order, k)])

                ret.append(mp)

        return ret

    def forecastAheadInterval(self, data, steps, **kwargs):

        l = len(data)

        ret = [[data[k], data[k]] for k in np.arange(l - self.order, l)]

        for k in np.arange(self.order, steps+self.order):

            if (len(self.transformations) > 0 and ret[-1][0] <= self.sets[0].lower and ret[-1][1] >= self.sets[
                -1].upper) or (len(self.transformations) == 0 and ret[-1][0] <= self.original_min and ret[-1][
                1] >= self.original_max):
                ret.append(ret[-1])
            else:
                lower = self.forecastInterval([ret[x][0] for x in np.arange(k - self.order, k)])
                upper = self.forecastInterval([ret[x][1] for x in np.arange(k - self.order, k)])

                ret.append([np.min(lower), np.max(upper)])

        return ret

    def forecastAheadDistribution(self, data, steps, **kwargs):

        ret = []

        method = kwargs.get('method', 2)
        smooth = "KDE" if method != 4 else "none"
        nbins = kwargs.get("num_bins", 100)

        uod = self.get_UoD()
        _bins = np.linspace(uod[0], uod[1], nbins).tolist()

        if method != 4:
            intervals = self.forecastAheadInterval(data, steps)
        else:
            l = len(data)
            for k in np.arange(l - self.order, l):
                dist = ProbabilityDistribution.ProbabilityDistribution(smooth, uod=uod, bins=_bins, **kwargs)
                dist.set(data[k], 1.0)
                ret.append(dist)

        for k in np.arange(self.order, steps + self.order):

            data = []

            if method == 1:

                lags = {}

                cc = 0

                for i in intervals[k - self.order : k]:

                    quantiles = []

                    for qt in np.arange(0, 50, 2):
                        quantiles.append(i[0] + qt * ((i[1] - i[0]) / 100))
                        quantiles.append(i[1] - qt * ((i[1] - i[0]) / 100))
                    quantiles.append(i[0] + ((i[1] - i[0]) / 2))

                    quantiles = list(set(quantiles))

                    quantiles.sort()

                    lags[cc] = quantiles

                    cc += 1

                # Build the tree with all possible paths

                root = tree.FLRGTreeNode(None)

                self.build_tree_without_order(root, lags, 0)

                # Trace the possible paths
                for p in root.paths():
                    path = list(reversed(list(filter(None.__ne__, p))))

                    qtle = np.ravel(self.forecastInterval(path))

                    data.extend(np.linspace(qtle[0],qtle[1],100).tolist())

            elif method == 2:

                for qt in np.arange(0, 50, 1):
                    # print(qt)
                    qtle_lower = self.forecastInterval(
                        [intervals[x][0] + qt * ((intervals[x][1] - intervals[x][0]) / 100) for x in
                         np.arange(k - self.order, k)])
                    qtle_lower = np.ravel(qtle_lower)
                    data.extend(np.linspace(qtle_lower[0], qtle_lower[1], 100).tolist())
                    qtle_upper = self.forecastInterval(
                        [intervals[x][1] - qt * ((intervals[x][1] - intervals[x][0]) / 100) for x in
                         np.arange(k - self.order, k)])
                    qtle_upper = np.ravel(qtle_upper)
                    data.extend(np.linspace(qtle_upper[0], qtle_upper[1], 100).tolist())
                qtle_mid = self.forecastInterval(
                    [intervals[x][0] + (intervals[x][1] - intervals[x][0]) / 2 for x in np.arange(k - self.order, k)])
                qtle_mid = np.ravel(qtle_mid)
                data.extend(np.linspace(qtle_mid[0], qtle_mid[1], 100).tolist())

            elif method == 3:
                i = intervals[k]

                data = np.linspace(i[0],i[1],100).tolist()

            else:
                dist = ProbabilityDistribution.ProbabilityDistribution(smooth, bins=_bins,
                                                                       uod=uod, **kwargs)
                lags = {}

                cc = 0

                for dd in ret[k - self.order: k]:
                    vals = [float(v) for v in dd.bins if round(dd.density(v),4) > 0]
                    lags[cc] = sorted(vals)
                    cc += 1

                root = tree.FLRGTreeNode(None)

                self.build_tree_without_order(root, lags, 0)

                # Trace the possible paths
                for p in root.paths():
                    path = list(reversed(list(filter(None.__ne__, p))))

                    pk = np.prod([ret[k - self.order + o].density(path[o])
                                  for o in np.arange(0,self.order)])

                    d = self.forecastDistribution(path)[0]

                    for bin in _bins:
                        dist.set(bin, dist.density(bin) + pk * d.density(bin))

            if method != 4:
                dist = ProbabilityDistribution.ProbabilityDistribution(smooth, bins=_bins, data=data,
                                                                       uod=uod, **kwargs)

            ret.append(dist)

        return ret

    def __str__(self):
        tmp = self.name + ":\n"
        for r in sorted(self.flrgs):
            p = round(self.flrgs[r].frequency_count / self.global_frequency_count, 3)
            tmp = tmp + "(" + str(p) + ") " + str(self.flrgs[r]) + "\n"
        return tmp
