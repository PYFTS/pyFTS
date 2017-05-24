#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
from pyFTS.common import FuzzySet,FLR
from pyFTS import hofts, fts, tree


class IntervalFTS(hofts.HighOrderFTS):
    """High Order Interval Fuzzy Time Series"""
    def __init__(self, name, **kwargs):
        super(IntervalFTS, self).__init__(order=1, name="IFTS " + name)
        self.shortname = "IFTS " + name
        self.name = "Interval FTS"
        self.detail = "Silva, P.; Guimarães, F.; Sadaei, H. (2016)"
        self.flrgs = {}
        self.has_point_forecasting = False
        self.has_interval_forecasting = True
        self.is_high_order = True

    def getUpper(self, flrg):
        if flrg.strLHS() in self.flrgs:
            tmp = self.flrgs[flrg.strLHS()]
            ret = max(np.array([self.setsDict[s].upper for s in tmp.RHS]))
        else:
            ret = flrg.LHS[-1].upper
        return ret

    def getLower(self, flrg):
        if flrg.strLHS() in self.flrgs:
            tmp = self.flrgs[flrg.strLHS()]
            ret = min(np.array([self.setsDict[s].lower for s in tmp.RHS]))
        else:
            ret = flrg.LHS[-1].lower
        return ret

    def getSequenceMembership(self, data, fuzzySets):
        mb = [fuzzySets[k].membership(data[k]) for k in np.arange(0, len(data))]
        return mb

    def buildTree(self, node, lags, level):
        if level >= self.order:
            return

        for s in lags[level]:
            node.appendChild(tree.FLRGTreeNode(s))

        for child in node.getChildren():
            self.buildTree(child, lags, level + 1)

    def forecastInterval(self, data, **kwargs):

        ndata = np.array(self.doTransformations(data))

        l = len(ndata)

        ret = []

        for k in np.arange(self.order - 1, l):

            affected_flrgs = []
            affected_flrgs_memberships = []

            up = []
            lo = []

            # Achar os conjuntos que tem pert > 0 para cada lag
            count = 0
            lags = {}
            if self.order > 1:
                subset = ndata[k - (self.order - 1): k + 1]

                for instance in subset:
                    mb = FuzzySet.fuzzyInstance(instance, self.sets)
                    tmp = np.argwhere(mb)
                    idx = np.ravel(tmp)  # flat the array

                    if idx.size == 0:  # the element is out of the bounds of the Universe of Discourse
                        if instance <= self.sets[0].lower:
                            idx = [0]
                        elif instance >= self.sets[-1].upper:
                            idx = [len(self.sets) - 1]
                        else:
                            raise Exception(instance)


                    lags[count] = idx
                    count = count + 1

                # Constrói uma árvore com todos os caminhos possíveis

                root = tree.FLRGTreeNode(None)

                self.buildTree(root, lags, 0)

                # Traça os possíveis caminhos e costrói as HOFLRG's

                for p in root.paths():
                    path = list(reversed(list(filter(None.__ne__, p))))
                    flrg = hofts.HighOrderFLRG(self.order)
                    for kk in path: flrg.appendLHS(self.sets[kk])

                    affected_flrgs.append(flrg)

                    # Acha a pertinência geral de cada FLRG
                    affected_flrgs_memberships.append(min(self.getSequenceMembership(subset, flrg.LHS)))
            else:

                mv = FuzzySet.fuzzyInstance(ndata[k], self.sets)
                tmp = np.argwhere(mv)
                idx = np.ravel(tmp)

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
                # achar o os bounds de cada FLRG, ponderados pela pertinência
                up.append(affected_flrgs_memberships[count] * self.getUpper(flrg))
                lo.append(affected_flrgs_memberships[count] * self.getLower(flrg))
                count = count + 1

            # gerar o intervalo
            norm = sum(affected_flrgs_memberships)
            lo_ = sum(lo) / norm
            up_ = sum(up) / norm
            ret.append([lo_, up_])

        ret = self.doInverseTransformations(ret, params=[data[self.order - 1:]], interval=True)

        return ret
