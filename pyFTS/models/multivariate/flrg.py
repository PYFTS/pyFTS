
import numpy as np
from pyFTS.common import flrg as flg


class FLRG(flg.FLRG):
    """
    Multivariate Fuzzy Logical Rule Group
    """

    def __init__(self, **kwargs):
        super(FLRG,self).__init__(0,**kwargs)
        self.order = kwargs.get('order', 1)
        self.LHS = kwargs.get('lhs', {})
        self.RHS = set()

    def set_lhs(self, var, fset):
        if self.order == 1:
            self.LHS[var] = fset
        else:
            if var not in self.LHS:
                self.LHS[var] = []
            self.LHS[var].append(fset)


    def append_rhs(self, fset, **kwargs):
        self.RHS.add(fset)

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