#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
import math
from operator import itemgetter
from pyFTS.common import FuzzySet, SortedCollection
from pyFTS import hofts, ifts, tree


class ProbabilisticFLRG(hofts.HighOrderFLRG):
    def __init__(self, order):
        super(ProbabilisticFLRG, self).__init__(order)
        self.RHS = {}
        self.frequencyCount = 0.0

    def appendRHS(self, c):
        self.frequencyCount += 1.0
        if c.name in self.RHS:
            self.RHS[c.name] += 1.0
        else:
            self.RHS[c.name] = 1.0

    def getProbability(self, c):
        return self.RHS[c] / self.frequencyCount

    def __str__(self):
        tmp2 = ""
        for c in sorted(self.RHS):
            if len(tmp2) > 0:
                tmp2 = tmp2 + ", "
            tmp2 = tmp2 + "(" + str(round(self.RHS[c] / self.frequencyCount, 3)) + ")" + c
        return self.strLHS() + " -> " + tmp2


class ProbabilisticFTS(ifts.IntervalFTS):
    def __init__(self, name):
        super(ProbabilisticFTS, self).__init__("PFTS")
        self.shortname = "PFTS " + name
        self.name = "Probabilistic FTS"
        self.detail = "Silva, P.; Guimarães, F.; Sadaei, H."
        self.flrgs = {}
        self.globalFrequency = 0
        self.hasPointForecasting = True
        self.hasIntervalForecasting = True
        self.hasDistributionForecasting = True
        self.isHighOrder = True

    def generateFLRG(self, flrs):
        flrgs = {}
        l = len(flrs)
        for k in np.arange(self.order, l+1):
            if self.dump: print("FLR: " + str(k))
            flrg = ProbabilisticFLRG(self.order)

            for kk in np.arange(k - self.order, k):
                flrg.appendLHS(flrs[kk].LHS)
                if self.dump: print("LHS: " + str(flrs[kk]))

            if flrg.strLHS() in flrgs:
                flrgs[flrg.strLHS()].appendRHS(flrs[k-1].RHS)
            else:
                flrgs[flrg.strLHS()] = flrg;
                flrgs[flrg.strLHS()].appendRHS(flrs[k-1].RHS)
            if self.dump: print("RHS: " + str(flrs[k-1]))

            self.globalFrequency += 1
        return (flrgs)

    def addNewPFLGR(self,flrg):
        if flrg.strLHS() not in self.flrgs:
            tmp = ProbabilisticFLRG(self.order)
            for fs in flrg.LHS: tmp.appendLHS(fs)
            tmp.appendRHS(flrg.LHS[-1])
            self.flrgs[tmp.strLHS()] = tmp;
            self.globalFrequency += 1

    def getProbability(self, flrg):
        if flrg.strLHS() in self.flrgs:
            return self.flrgs[flrg.strLHS()].frequencyCount / self.globalFrequency
        else:
            self.addNewPFLGR(flrg)
            return self.getProbability(flrg)

    def getMidpoints(self, flrg):
        if flrg.strLHS() in self.flrgs:
            tmp = self.flrgs[flrg.strLHS()]
            ret = sum(np.array([tmp.getProbability(s) * self.setsDict[s].centroid for s in tmp.RHS]))
        else:
            pi = 1 / len(flrg.LHS)
            ret = sum(np.array([pi * s.centroid for s in flrg.LHS]))
        return ret

    def getUpper(self, flrg):
        if flrg.strLHS() in self.flrgs:
            tmp = self.flrgs[flrg.strLHS()]
            ret = sum(np.array([tmp.getProbability(s) * self.setsDict[s].upper for s in tmp.RHS]))
        else:
            pi = 1 / len(flrg.LHS)
            ret = sum(np.array([pi * s.upper for s in flrg.LHS]))
        return ret

    def getLower(self, flrg):
        if flrg.strLHS() in self.flrgs:
            tmp = self.flrgs[flrg.strLHS()]
            ret = sum(np.array([tmp.getProbability(s) * self.setsDict[s].lower for s in tmp.RHS]))
        else:
            pi = 1 / len(flrg.LHS)
            ret = sum(np.array([pi * s.lower for s in flrg.LHS]))
        return ret

    def forecast(self, data):

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
            count = 0
            lags = {}
            if self.order > 1:
                subset = ndata[k - (self.order - 1): k + 1]

                for instance in subset:
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
                    count = count + 1

                # Build the tree with all possible paths

                root = tree.FLRGTreeNode(None)

                self.buildTree(root, lags, 0)

                # Trace the possible paths and build the PFLRG's

                for p in root.paths():
                    path = list(reversed(list(filter(None.__ne__, p))))
                    flrg = hofts.HighOrderFLRG(self.order)
                    for kk in path: flrg.appendLHS(self.sets[kk])

                    assert len(flrg.LHS) == subset.size, str(subset) + " -> " + str([s.name for s in flrg.LHS])

                    ##
                    affected_flrgs.append(flrg)

                    # Find the general membership of FLRG
                    affected_flrgs_memberships.append(min(self.getSequenceMembership(subset, flrg.LHS)))

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

            count = 0
            for flrg in affected_flrgs:
                # achar o os bounds de cada FLRG, ponderados pela probabilidade e pertinência
                norm = self.getProbability(flrg) * affected_flrgs_memberships[count]
                if norm == 0:
                    norm = self.getProbability(flrg)  # * 0.001
                mp.append(norm * self.getMidpoints(flrg))
                norms.append(norm)
                count = count + 1

            # gerar o intervalo
            norm = sum(norms)
            if norm == 0:
                ret.append(0)
            else:
                ret.append(sum(mp) / norm)

        ret = self.doInverseTransformations(ret, params=[data[self.order - 1:]])

        return ret

    def forecastInterval(self, data):

        data = np.array(data)

        ndata = self.doTransformations(data)

        l = len(ndata)

        ret = []

        for k in np.arange(self.order - 1, l):

            # print(k)

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
                        if instance <= self.sets[0].lower:
                            idx = [0]
                        elif instance >= self.sets[-1].upper:
                            idx = [len(self.sets) - 1]
                        else:
                            raise Exception(instance)

                    lags[count] = idx
                    count = count + 1

                # Build the tree with all possible paths

                root = tree.FLRGTreeNode(None)

                self.buildTree(root, lags, 0)

                # Trace the possible paths and build the PFLRG's

                for p in root.paths():
                    path = list(reversed(list(filter(None.__ne__, p))))
                    flrg = hofts.HighOrderFLRG(self.order)
                    for kk in path: flrg.appendLHS(self.sets[kk])

                    assert len(flrg.LHS) == subset.size, str(subset) + " -> " + str([s.name for s in flrg.LHS])

                    ##
                    affected_flrgs.append(flrg)

                    # Find the general membership of FLRG
                    affected_flrgs_memberships.append(min(self.getSequenceMembership(subset, flrg.LHS)))

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

            count = 0
            for flrg in affected_flrgs:
                # achar o os bounds de cada FLRG, ponderados pela probabilidade e pertinência
                norm = self.getProbability(flrg) * affected_flrgs_memberships[count]
                if norm == 0:
                    norm = self.getProbability(flrg)  # * 0.001
                up.append(norm * self.getUpper(flrg))
                lo.append(norm * self.getLower(flrg))
                norms.append(norm)
                count = count + 1

            # gerar o intervalo
            norm = sum(norms)
            if norm == 0:
                ret.append([0, 0])
            else:
                lo_ = self.doInverseTransformations(sum(lo) / norm, params=[data[k - (self.order - 1): k + 1]])
                up_ = self.doInverseTransformations(sum(up) / norm, params=[data[k - (self.order - 1): k + 1]])
                ret.append([lo_, up_])

        return ret

    def forecastAhead(self, data, steps):
        ret = [data[k] for k in np.arange(len(data) - self.order, len(data))]

        for k in np.arange(self.order - 1, steps):

            if ret[-1] <= self.sets[0].lower or ret[-1] >= self.sets[-1].upper:
                ret.append(ret[-1])
            else:
                mp = self.forecast([ret[x] for x in np.arange(k - self.order, k)])

                ret.append(mp)

        return ret

    def forecastAheadInterval(self, data, steps):
        ret = [[data[k], data[k]] for k in np.arange(len(data) - self.order, len(data))]

        for k in np.arange(self.order, steps+self.order):

            if ret[-1][0] <= self.sets[0].lower and ret[-1][1] >= self.sets[-1].upper:
                ret.append(ret[-1])
            else:
                lower = self.forecastInterval([ret[x][0] for x in np.arange(k - self.order, k)])
                upper = self.forecastInterval([ret[x][1] for x in np.arange(k - self.order, k)])

                ret.append([np.min(lower), np.max(upper)])

        return ret

    def getGridClean(self, resolution):
        grid = {}
        for sbin in np.arange(self.sets[0].lower, self.sets[-1].upper, resolution):
            grid[sbin] = 0

        return grid

    def gridCount(self, grid, resolution, index, interval):
        #print(interval)
        for k in index.inside(interval[0],interval[1]):
            #print(k)
            grid[k] += 1
        return grid

    def gridCountPoint(self, grid, resolution, index, point):
        k = index.find_ge(point)
        # print(k)
        grid[k] += 1
        return grid

    def buildTreeWithoutOrder(self, node, lags, level):

        if level not in lags:
            return

        for s in lags[level]:
            node.appendChild(tree.FLRGTreeNode(s))

        for child in node.getChildren():
            self.buildTreeWithoutOrder(child, lags, level + 1)

    def forecastAheadDistribution(self, data, steps, resolution,parameters=None):

        ret = []

        intervals = self.forecastAheadInterval(data, steps)

        grid = self.getGridClean(resolution)

        index = SortedCollection.SortedCollection(iterable=grid.keys())

        grids = []
        for k in np.arange(0, steps):
            grids.append(self.getGridClean(resolution))

        for k in np.arange(self.order, steps + self.order):

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

            self.buildTreeWithoutOrder(root, lags, 0)

            # Trace the possible paths

            for p in root.paths():
                path = list(reversed(list(filter(None.__ne__, p))))

                if parameters is None:
                    qtle = self.forecastInterval(path)
                    grids[k - self.order] = self.gridCount(grids[k - self.order], resolution, index, np.ravel(qtle))
                else:
                    qtle = self.forecast(path)
                    grids[k - self.order] = self.gridCountPoint(grids[k - self.order], resolution, index, np.ravel(qtle))

        for k in np.arange(0, steps):
            tmp = np.array([grids[k][q] for q in sorted(grids[k])])
            ret.append(tmp / sum(tmp))

        grid = self.getGridClean(resolution)
        df = pd.DataFrame(ret, columns=sorted(grid))
        return df


    def __str__(self):
        tmp = self.name + ":\n"
        for r in sorted(self.flrgs):
            p = round(self.flrgs[r].frequencyCount / self.globalFrequency, 3)
            tmp = tmp + "(" + str(p) + ") " + str(self.flrgs[r]) + "\n"
        return tmp
