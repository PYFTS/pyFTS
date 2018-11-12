from pyFTS.partitioners import partitioner
from pyFTS.models.multivariate.common import MultivariateFuzzySet
from itertools import product

class GridCluster(partitioner.Partitioner):
    """
    A cartesian product of all fuzzy sets of all variables
    """

    def __init__(self, **kwargs):
        super(GridCluster, self).__init__(name="GridCluster", preprocess=False, **kwargs)

        self.mvfts = kwargs.get('mvfts', None)
        self.sets = {}
        self.build(None)

    def build(self, data):
        fsets = [[x for x in k.partitioner.sets.values()]
                 for k in self.mvfts.explanatory_variables]

        c = 0
        for k in product(*fsets):
            key = self.prefix+str(c)
            mvfset = MultivariateFuzzySet(name=key)
            c += 1
            for fset in k:
                mvfset.append_set(fset.variable, fset)
            self.sets[key] = mvfset

