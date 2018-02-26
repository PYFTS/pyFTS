import numpy as np


class FLRG(object):

    def __init__(self, order, **kwargs):
        self.LHS = None
        self.RHS = None
        self.order = order
        self.midpoint = None
        self.lower = None
        self.upper = None

    def get_membership(self, data):
        ret = 0.0
        if isinstance(self.LHS, (list, set)):
            assert len(self.LHS) == len(data)
            ret = min([self.LHS[ct].membership(dat) for ct, dat in enumerate(data)])
        else:
            ret = self.LHS.membership(data)
        return ret

    def get_midpoint(self):
        if self.midpoint is None:
            self.midpoint = sum(self.get_midpoints())/len(self.RHS)
        return self.midpoint

    def get_midpoints(self):
        if isinstance(self.RHS, (list, set)):
            return np.array([s.centroid for s in self.RHS])
        elif isinstance(self.RHS, dict):
            return np.array([self.RHS[s].centroid for s in self.RHS.keys()])

    def get_lower(self):
        if self.lower is None:
            if isinstance(self.RHS, list):
                self.lower = min([rhs.lower for rhs in self.RHS])
            elif isinstance(self.RHS, dict):
                self.lower = min([self.RHS[s].lower for s in self.RHS.keys()])
        return self.lower

    def get_upper(self, t):
        if self.upper is None:
            if isinstance(self.RHS, list):
                self.upper = max([rhs.upper for rhs in self.RHS])
            elif isinstance(self.RHS, dict):
                self.upper = max([self.RHS[s].upper for s in self.RHS.keys()])
        return self.upper

    def __len__(self):
        return len(self.RHS)



