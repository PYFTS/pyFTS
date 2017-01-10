import numpy as np
import pandas as pd
import math
from pyFTS.common import FuzzySet, FLR
from pyFTS import hofts, ifts, tree


class ProbabilisticFLRG(hofts.HighOrderFLRG):
    def __init__(self, order):
        super(ProbabilisticFLRG, self).__init__(order)
        self.RHS = {}
        self.frequencyCount = 0.0

    def appendRHS(self, c):
        self.frequencyCount += 1
        if c.name in self.RHS:
            self.RHS[c.name] += 1
        else:
            self.RHS[c.name] = 1.0

    def getProbability(self, c):
        return self.RHS[c] / self.frequencyCount

    def __str__(self):
        tmp2 = ""
        for c in sorted(self.RHS):
            if len(tmp2) > 0:
                tmp2 = tmp2 + ", "
            tmp2 = tmp2 + c + "(" + str(round(self.RHS[c] / self.frequencyCount, 3)) + ")"
        return self.strLHS() + " -> " + tmp2


class ProbabilisticFTS(ifts.IntervalFTS):
    def __init__(self, name):
        super(ProbabilisticFTS, self).__init__("PIFTS")
        self.shortname = "PIFTS " + name
        self.name = "Probabilistic FTS"
        self.detail = "Silva, P.; Guimarães, F.; Sadaei, H."
        self.flrgs = {}
        self.globalFrequency = 0
        self.hasPointForecasting = True
        self.hasIntervalForecasting = True
        self.hasDistributionForecasting = True

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

            self.globalFrequency = self.globalFrequency + 1
        return (flrgs)

    def getProbability(self, flrg):
        if flrg.strLHS() in self.flrgs:
            return self.flrgs[flrg.strLHS()].frequencyCount / self.globalFrequency
        else:
            return 1.0 / self.globalFrequency

    def getMidpoints(self, flrg):
        if flrg.strLHS() in self.flrgs:
            tmp = self.flrgs[flrg.strLHS()]
            ret = sum(np.array([tmp.getProbability(s) * self.setsDict[s].centroid for s in tmp.RHS]))
        else:
            ret = sum(np.array([0.33 * s.centroid for s in flrg.LHS]))
        return ret

    def getUpper(self, flrg):
        if flrg.strLHS() in self.flrgs:
            tmp = self.flrgs[flrg.strLHS()]
            ret = sum(np.array([tmp.getProbability(s) * self.setsDict[s].upper for s in tmp.RHS]))
        else:
            ret = sum(np.array([0.33 * s.upper for s in flrg.LHS]))
        return ret

    def getLower(self, flrg):
        if flrg.strLHS() in self.flrgs:
            tmp = self.flrgs[flrg.strLHS()]
            ret = sum(np.array([tmp.getProbability(s) * self.setsDict[s].lower for s in tmp.RHS]))
        else:
            ret = sum(np.array([0.33 * s.lower for s in flrg.LHS]))
        return ret

    def forecast(self, data):

        ndata = np.array(data)

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
                        if math.ceil(instance) <= self.sets[0].lower:
                            idx = [0]
                        elif math.ceil(instance) >= self.sets[-1].upper:
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
                    if math.ceil(ndata[k]) <= self.sets[0].lower:
                        idx = [0]
                    elif math.ceil(ndata[k]) >= self.sets[-1].upper:
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
                ret.append([0, 0])
            else:
                ret.append(sum(mp) / norm)

        return ret

    def forecastInterval(self, data):

        ndata = np.array(data)

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
                        if math.ceil(instance) <= self.sets[0].lower:
                            idx = [0]
                        elif math.ceil(instance) >= self.sets[-1].upper:
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
                    if math.ceil(ndata[k]) <= self.sets[0].lower:
                        idx = [0]
                    elif math.ceil(ndata[k]) >= self.sets[-1].upper:
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
                ret.append([sum(lo) / norm, sum(up) / norm])

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

        for k in np.arange(self.order - 1, steps):

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

    def gridCount(self, grid, resolution, interval):
        for sbin in sorted(grid):
            if sbin >= interval[0] and (sbin + resolution) <= interval[1]:
                grid[sbin] = grid[sbin] + 1
        return grid

    def forecastDistributionAhead2(self, data, steps, resolution):

        ret = []

        intervals = self.forecastAhead(data, steps)

        for k in np.arange(self.order, steps):

            grid = self.getGridClean(resolution)
            grid = self.gridCount(grid, resolution, intervals[k])

            lags = {}

            cc = 0
            for x in np.arange(k - self.order, k):
                tmp = []
                for qt in np.arange(0, 100, 5):
                    tmp.append(intervals[x][0] + qt * (intervals[x][1] - intervals[x][0]) / 100)
                    tmp.append(intervals[x][1] - qt * (intervals[x][1] - intervals[x][0]) / 100)
                tmp.append(intervals[x][0] + (intervals[x][1] - intervals[x][0]) / 2)

                lags[cc] = tmp

                cc = cc + 1
            # Build the tree with all possible paths

            root = tree.FLRGTreeNode(None)

            self.buildTree(root, lags, 0)

            # Trace the possible paths and build the PFLRG's

            for p in root.paths():
                path = list(reversed(list(filter(None.__ne__, p))))

                subset = [kk for kk in path]

                qtle = self.forecast(subset)
                grid = self.gridCount(grid, resolution, np.ravel(qtle))

            tmp = np.array([grid[k] for k in sorted(grid)])
            ret.append(tmp / sum(tmp))

        grid = self.getGridClean(resolution)
        df = pd.DataFrame(ret, columns=sorted(grid))
        return df

    def forecastAheadDistribution(self, data, steps, resolution):

        ret = []

        intervals = self.forecastAheadInterval(data, steps)

        for k in np.arange(self.order, steps):

            grid = self.getGridClean(resolution)
            grid = self.gridCount(grid, resolution, intervals[k])

            for qt in np.arange(1, 50, 2):
                # print(qt)
                qtle_lower = self.forecastInterval(
                    [intervals[x][0] + qt * (intervals[x][1] - intervals[x][0]) / 100 for x in
                     np.arange(k - self.order, k)])
                grid = self.gridCount(grid, resolution, np.ravel(qtle_lower))
                qtle_upper = self.forecastInterval(
                    [intervals[x][1] - qt * (intervals[x][1] - intervals[x][0]) / 100 for x in
                     np.arange(k - self.order, k)])
                grid = self.gridCount(grid, resolution, np.ravel(qtle_upper))
            qtle_mid = self.forecastInterval(
                [intervals[x][0] + (intervals[x][1] - intervals[x][0]) / 2 for x in np.arange(k - self.order, k)])
            grid = self.gridCount(grid, resolution, np.ravel(qtle_mid))

            tmp = np.array([grid[k] for k in sorted(grid)])

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
