
from pyFTS.common import flrg
from pyFTS.models.nonstationary import common


class NonStationaryFLRG(flrg.FLRG):

    def __init__(self, LHS, **kwargs):
        super(NonStationaryFLRG, self).__init__(1, **kwargs)
        self.LHS = LHS
        self.RHS = set()

    def get_key(self):
        return self.LHS.name

    def get_membership(self, data, t, window_size=1):
        ret = 0.0
        if isinstance(self.LHS, (list, set)):
            #assert len(self.LHS) == len(data)

            ret = min([self.LHS[ct].membership(dat, common.window_index(t - (self.order - ct), window_size))
                       for ct, dat in enumerate(data)])
        else:
            ret = self.LHS.membership(data, common.window_index(t, window_size))
        return ret

    def get_midpoint(self, t, window_size=1):
        if len(self.RHS) > 0:
            if isinstance(self.RHS, (list, set)):
                tmp = [r.get_midpoint(common.window_index(t, window_size)) for r in self.RHS]
            elif isinstance(self.RHS, dict):
                tmp = [self.RHS[r].get_midpoint(common.window_index(t, window_size)) for r in self.RHS.keys()]
            return sum(tmp) / len(tmp)
        else:
            return self.LHS[-1].get_midpoint(common.window_index(t, window_size))

    def get_lower(self, t, window_size=1):
        if len(self.RHS) > 0:
            if isinstance(self.RHS, (list, set)):
                return min([r.get_lower(common.window_index(t, window_size)) for r in self.RHS])
            elif isinstance(self.RHS, dict):
                return min([self.RHS[r].get_lower(common.window_index(t, window_size)) for r in self.RHS.keys()])
        else:
            return self.LHS[-1].get_lower(common.window_index(t, window_size))

    def get_upper(self, t, window_size=1):
        if len(self.RHS) > 0:
            if isinstance(self.RHS, (list, set)):
                return max([r.get_upper(common.window_index(t, window_size)) for r in self.RHS])
            elif isinstance(self.RHS, dict):
                return max([self.RHS[r].get_upper(common.window_index(t, window_size)) for r in self.RHS.keys()])
        else:
            return self.LHS[-1].get_upper(common.window_index(t, window_size))
