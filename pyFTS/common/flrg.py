import numpy as np


class FLRG(object):

    def __init__(self, order, **kwargs):
        self.LHS = None
        self.RHS = None
        self.order = order
        self.midpoint = None
        self.lower = None
        self.upper = None
        self.key = None

    def append_rhs(self, set, **kwargs):
        pass

    def get_key(self):
        if self.key is None:
            if isinstance(self.LHS, (list, set)):
                names = [c for c in self.LHS]
            elif isinstance(self.LHS, dict):
                names = [self.LHS[k] for k in self.LHS.keys()]
            else:
                names = [self.LHS]

            self.key = ""

            for n in names:
                if len(self.key) > 0:
                    self.key += ","
                self.key = self.key + n
        return self.key


    def get_membership(self, data, sets):
        ret = 0.0
        if isinstance(self.LHS, (list, set)):
            assert len(self.LHS) == len(data)
            ret = np.nanmin([sets[self.LHS[ct]].membership(dat) for ct, dat in enumerate(data)])
        else:
            ret = sets[self.LHS].membership(data)
        return ret

    def get_midpoint(self, sets):
        if self.midpoint is None:
            self.midpoint = np.nanmean(self.get_midpoints(sets))
        return self.midpoint

    def get_midpoints(self, sets):
        if isinstance(self.RHS, (list, set)):
            return np.array([sets[s].centroid for s in self.RHS])
        elif isinstance(self.RHS, dict):
            return np.array([sets[self.RHS[s]].centroid for s in self.RHS.keys()])

    def get_lower(self, sets):
        if self.lower is None:
            if isinstance(self.RHS, list):
                self.lower = min([sets[rhs].lower for rhs in self.RHS])
            elif isinstance(self.RHS, dict):
                self.lower = min([sets[self.RHS[s]].lower for s in self.RHS.keys()])
        return self.lower

    def get_upper(self, sets):
        if self.upper is None:
            if isinstance(self.RHS, list):
                self.upper = max([sets[rhs].upper for rhs in self.RHS])
            elif isinstance(self.RHS, dict):
                self.upper = max([sets[self.RHS[s]].upper for s in self.RHS.keys()])
        return self.upper

    def __len__(self):
        return len(self.RHS)



