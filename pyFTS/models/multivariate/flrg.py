
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
        self.key = None

    def set_lhs(self, var, set):
        self.LHS[var] = set

    def append_rhs(self, set):
        self.RHS.add(set)

    def get_key(self):
        if self.key is None:
            _str = ""
            for k in self.LHS.keys():
                _str += "," if len(_str) > 0 else ""
                _str += k + ":" + self.LHS[k].name
            self.key = _str

        return self.key

    def get_membership(self, data):
        return np.nanmin([self.LHS[k].membership(data[k]) for k in self.LHS.keys()])

    def __str__(self):
        _str = ""
        for k in self.RHS:
            _str += "," if len(_str) > 0 else ""
            _str += k.name

        return self.get_key() + " -> " + _str