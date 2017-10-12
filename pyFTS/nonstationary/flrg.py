
from pyFTS import flrg
from pyFTS.nonstationary import common


class NonStationaryFLRG(flrg.FLRG):

    def __init__(self, LHS, **kwargs):
        super(NonStationaryFLRG, self).__init__(1, **kwargs)
        self.LHS = LHS
        self.RHS = set()

    def get_membership(self, data, t, window_size=1):
        ret = 0.0
        if isinstance(self.LHS, (list, set)):
            assert len(self.LHS) == len(data)

            ret = min([self.LHS[ct].membership(dat, common.window_index(t - (self.order - ct), window_size))
                       for ct, dat in enumerate(data)])
        else:
            ret = self.LHS.membership(data, common.window_index(t, window_size))
        return ret

    def get_midpoint(self, t, window_size=1):
        if len(self.RHS) > 0:
            if isinstance(self.RHS, (list,set)):
                tmp = [r.get_midpoint(common.window_index(t, window_size)) for r in self.RHS]
            elif isinstance(self.RHS, dict):
                tmp = [self.RHS[r].get_midpoint(common.window_index(t, window_size)) for r in self.RHS.keys()]
            return sum(tmp) / len(tmp)
        else:
            return self.LHS[-1].get_midpoint(common.window_index(t, window_size))


    def get_lower(self, t, window_size=1):
        if self.lower is None:
            if len(self.RHS) > 0:
                self.lower = min([r.get_lower(common.window_index(t, window_size)) for r in self.RHS])
            else:
                self.lower = self.LHS[-1].get_lower(common.window_index(t, window_size))

        return self.lower

    def get_upper(self, t, window_size=1):
        if self.upper is None:
            if len(self.RHS) > 0:
                self.upper = min([r.get_upper(common.window_index(t, window_size)) for r in self.RHS])
            else:
                self.upper = self.LHS[-1].get_upper(common.window_index(t, window_size))
        return self.upper