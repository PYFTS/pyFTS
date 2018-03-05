
import numpy as np
from pyFTS.common import flrg as flg


class FLRG(flg.FLRG):
    """
    Multivariate Fuzzy Logical Rule Group
    """

    def __init__(self, **kwargs):
        super(FLRG,self).__init__(0,**kwargs)
        self.LHS = kwargs.get('lhs', {})
        self.RHS = set()

    def set_lhs(self, var, set):
        self.LHS[var] = set

    def append_rhs(self, set, **kwargs):
        self.RHS.add(set)

    def get_membership(self, data, variables):
        mvs = []
        for var in variables:
            s = self.LHS[var.name]
            mvs.append(var.partitioner.sets[s].membership(data[var.name]))

        return np.nanmin(mvs)

    def __str__(self):
        _str = ""
        for k in self.RHS:
            _str += "," if len(_str) > 0 else ""
            _str += k

        return self.get_key() + " -> " + _str