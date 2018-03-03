

class FLR(object):
    """Multivariate Fuzzy Logical Relationship"""

    def __init__(self):
        """
        Creates a Fuzzy Logical Relationship
        :param LHS: Left Hand Side fuzzy set
        :param RHS: Right Hand Side fuzzy set
        """
        self.LHS = {}
        self.RHS = None

    def set_lhs(self, var, set):
        self.LHS[var] = set

    def set_rhs(self, set):
        self.RHS = set

    def __str__(self):
        return str([self.LHS[k] for k in self.LHS.keys()]) + " -> " + self.RHS



