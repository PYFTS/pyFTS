
from pyFTS import flrg


class NonStationaryFLRG(flrg.FLRG):

    def __init__(self, LHS, **kwargs):
        super(NonStationaryFLRG, self).__init__(1, **kwargs)
        self.LHS = LHS
        self.RHS = set()

    def get_midpoint(self, t):
        if self.midpoint is None:
            tmp = [r.get_midpoint(t) for r in self.RHS]
            self.midpoint = sum(tmp) / len(tmp)
        return self.midpoint

    def get_lower(self, t):
        if self.lower is None:
            self.lower = min([r.get_lower(t) for r in self.RHS])
        return self.lower

    def get_upper(self, t):
        if self.upper is None:
            self.upper = min([r.get_upper(t) for r in self.RHS])
        return self.upper