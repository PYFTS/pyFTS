
from pyFTS import flrg


class NonStationaryFLRG(flrg.FLRG):

    def __init__(self, LHS, **kwargs):
        super(NonStationaryFLRG, self).__init__(1, **kwargs)
        self.LHS = LHS
        self.RHS = set()

    def get_midpoint(self, t):
        if self.midpoint is None:
            tmp = []
            for r in self.RHS:
                tmp.append(r.get_midpoint(t))
            self.midpoint = sum(tmp) / len(tmp)
        return self.midpoint

    def get_lower(self, t):
        if self.lower is None:
            tmp = []
            for r in self.RHS:
                tmp.append(r.get_midpoint(t))
            self.lower = min(tmp)
        return self.lower

    def get_upper(self, t):
        if self.upper is None:
            tmp = []
            for r in self.RHS:
                tmp.append(r.get_midpoint(t))
            self.upper = max(tmp)
        return self.upper