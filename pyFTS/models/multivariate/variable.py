from pyFTS.common import fts, FuzzySet, FLR, Membership, tree
from pyFTS.partitioners import Grid
from pyFTS.models.multivariate import FLR as MVFLR


class Variable:
    def __init__(self, name, **kwargs):
        self.name = name
        self.alias = kwargs.get('alias', self.name)
        self.data_label = kwargs.get('data_label', self.name)
        self.type = kwargs.get('type', 'common')
        self.transformation = kwargs.get('transformation', None)
        self.transformation_params = kwargs.get('transformation_params', None)
        self.partitioner = None

        if kwargs.get('data', None) is not None:
            self.build(**kwargs)

    def build(self, **kwargs):
        fs = kwargs.get('partitioner', Grid.GridPartitioner)
        mf = kwargs.get('func', Membership.trimf)
        np = kwargs.get('npart', 10)
        data = kwargs.get('data', None)
        self.partitioner = fs(data=data[self.data_label].values, npart=np, func=mf,
                              transformation=self.transformation, prefix=self.alias,
                              variable=self.name)

    def apply_transformations(self, data, **kwargs):

        if kwargs.get('params', None) is not None:
            self.transformation_params = kwargs.get('params', None)

        if self.transformation is not None:
            return self.transformation.apply(data, self.transformation_params)

        return data

    def apply_inverse_transformations(self, data, **kwargs):

        if kwargs.get('params', None) is not None:
            self.transformation_params = kwargs.get('params', None)

        if self.transformation is not None:
            return self.transformation.inverse(data, self.transformation_params)

        return data

    def __str__(self):
        return self.name
