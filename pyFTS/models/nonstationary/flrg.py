
from pyFTS.common import flrg
from pyFTS.models.nonstationary import common
import numpy as np


class NonStationaryFLRG(flrg.FLRG):
    def unpack_args(self, *args):
        l = len(args)
        tmp = args
        sets, t, w = None, 0, 1
        if l > 0 and isinstance(tmp[0], dict):
            sets = tmp[0]
        if l > 1 and isinstance(tmp[1], (int, list, tuple)):
            t = tmp[1]
        if l > 2 and isinstance(tmp[2], int):
            w = tmp[2]

        return (sets, t, w)


    def __init__(self, LHS, **kwargs):
        super(NonStationaryFLRG, self).__init__(1, **kwargs)
        self.LHS = LHS
        self.RHS = set()

    def get_key(self):
        if isinstance(self.LHS, list):
            return str([k for k in self.LHS])
        elif isinstance(self.LHS, dict):
            return str(self.LHS.keys())
        else:
            return self.LHS

    def get_membership(self, data, *args):
        sets, t, window_size = self.unpack_args(*args)
        ret = 0.0
        if isinstance(self.LHS, (list, set)):
            ret = min([sets[self.LHS[ct]].membership(dat, common.window_index(t - (self.order - ct), window_size))
                       for ct, dat in enumerate(data)])
        else:
            ret = self.LHS.membership(data, common.window_index(t, window_size))
        return ret

    def get_midpoint(self, *args):
        sets, t, window_size = self.unpack_args(*args)
        if len(self.RHS) > 0:
            if isinstance(self.RHS, (list, set)):
                tmp = [sets[r].get_midpoint(common.window_index(t, window_size)) for r in self.RHS]
            elif isinstance(self.RHS, dict):
                tmp = [sets[r].get_midpoint(common.window_index(t, window_size)) for r in self.RHS.keys()]
            return sum(tmp) / len(tmp)
        else:
            return sets[self.LHS[-1]].get_midpoint(common.window_index(t, window_size))

    def get_lower(self, *args):
        sets, t, window_size = self.unpack_args(*args)
        if len(self.RHS) > 0:
            if isinstance(self.RHS, (list, set)):
                return min([sets[r].get_lower(common.window_index(t, window_size)) for r in self.RHS])
            elif isinstance(self.RHS, dict):
                return min([sets[r].get_lower(common.window_index(t, window_size)) for r in self.RHS.keys()])
        else:
            return sets[self.LHS[-1]].get_lower(common.window_index(t, window_size))

    def get_upper(self, *args):
        sets, t, window_size = self.unpack_args(*args)
        if len(self.RHS) > 0:
            if isinstance(self.RHS, (list, set)):
                return max([sets[r].get_upper(common.window_index(t, window_size)) for r in self.RHS])
            elif isinstance(self.RHS, dict):
                return max([sets[r].get_upper(common.window_index(t, window_size)) for r in self.RHS.keys()])
        else:
            return sets[self.LHS[-1]].get_upper(common.window_index(t, window_size))
