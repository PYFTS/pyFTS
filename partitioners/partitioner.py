from pyFTS.common import FuzzySet, Membership
import  numpy as np

class Partitioner(object):
    def __init__(self,name,data,npart,func = Membership.trimf, names=None, prefix="A"):
        self.name = name
        self.partitions = npart
        self.sets = []
        self.membership_function = func
        self.setnames = names
        self.prefix = prefix
        _min = min(data)
        if _min < 0:
            self.min = _min * 1.1
        else:
            self.min = _min * 0.9

        _max = max(data)
        if _max > 0:
            self.max = _max * 1.1
        else:
            self.max = _max * 0.9
        self.sets = self.build(data)

    def build(self,data):
        pass

    def plot(self,ax):
        ax.set_title(self.name)
        ax.set_ylim([0, 1])
        ax.set_xlim([self.min, self.max])
        for s in self.sets:
            if s.mf == Membership.trimf:
                ax.plot([s.parameters[0], s.parameters[1], s.parameters[2]], [0, 1, 0])
            elif s.mf == Membership.gaussmf:
                tmpx = [kk for kk in np.arange(s.lower, s.upper)]
                tmpy = [s.membership(kk) for kk in np.arange(s.lower, s.upper)]
                ax.plot(tmpx, tmpy)