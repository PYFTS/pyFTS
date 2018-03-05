#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
from pyFTS.common import FuzzySet, FLR, fts, tree
from pyFTS.models import hofts


class IntervalFTS(hofts.HighOrderFTS):
    """
    High Order Interval Fuzzy Time Series

    SILVA, Petrônio CL; SADAEI, Hossein Javedani; GUIMARÃES, Frederico Gadelha. Interval Forecasting with Fuzzy Time Series.
    In: Computational Intelligence (SSCI), 2016 IEEE Symposium Series on. IEEE, 2016. p. 1-8.
    """
    def __init__(self, name, **kwargs):
        super(IntervalFTS, self).__init__(name="IFTS " + name, **kwargs)
        self.shortname = "IFTS " + name
        self.name = "Interval FTS"
        self.detail = "Silva, P.; Guimarães, F.; Sadaei, H. (2016)"
        self.flrgs = {}
        self.has_point_forecasting = False
        self.has_interval_forecasting = True
        self.is_high_order = True

    def get_upper(self, flrg):
        if flrg.get_key() in self.flrgs:
            tmp = self.flrgs[flrg.get_key()]
            ret = tmp.get_upper(self.sets)
        else:
            ret = self.sets[flrg.LHS[-1]].upper
        return ret

    def get_lower(self, flrg):
        if flrg.get_key() in self.flrgs:
            tmp = self.flrgs[flrg.get_key()]
            ret = tmp.get_lower(self.sets)
        else:
            ret = self.sets[flrg.LHS[-1]].lower
        return ret

    def get_sequence_membership(self, data, fuzzySets):
        mb = [fuzzySets[k].membership(data[k]) for k in np.arange(0, len(data))]
        return mb


    def forecast_interval(self, data, **kwargs):

        ret = []

        l = len(data)

        if l <= self.order:
            return data

        ndata = self.apply_transformations(data)

        for k in np.arange(self.order, l+1):

            sample = ndata[k - self.order: k]

            flrgs = self.generate_lhs_flrg(sample)

            up = []
            lo = []
            affected_flrgs_memberships = []

            for flrg in flrgs:
                # achar o os bounds de cada FLRG, ponderados pela pertinência
                mv = flrg.get_membership(sample, self.sets)
                up.append(mv * self.get_upper(flrg))
                lo.append(mv * self.get_lower(flrg))
                affected_flrgs_memberships.append(mv)

            # gerar o intervalo
            norm = sum(affected_flrgs_memberships)
            lo_ = sum(lo) / norm
            up_ = sum(up) / norm
            ret.append([lo_, up_])

        ret = self.apply_inverse_transformations(ret, params=[data[self.order - 1:]], interval=True)

        return ret
