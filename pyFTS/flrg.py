

class FLRG(object):

    def __init__(self, order, **kwargs):
        self.LHS = None
        self.RHS = None
        self.order = order
        self.midpoint = None
        self.lower = None
        self.upper = None

    def __len__(self):
        return len(self.RHS)



