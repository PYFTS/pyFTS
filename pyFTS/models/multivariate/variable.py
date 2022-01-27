import pandas as pd
from pyFTS.common import fts, FuzzySet, FLR, Membership, tree
from pyFTS.partitioners import Grid
from pyFTS.models.multivariate import FLR as MVFLR


class Variable:
    """
    A variable of a fuzzy time series multivariate model. Each variable contains its own
    transformations and partitioners.
    """
    def __init__(self, name, **kwargs):
        """

        :param name:
        :param \**kwargs: See below

        :Keyword Arguments:
            * *alias* -- Alternative name for the variable
        """
        self.name = name
        """A string with the name of the variable"""
        self.alias = kwargs.get('alias', self.name)
        """A string with the alias of the variable"""
        self.data_label = kwargs.get('data_label', self.name)
        """A string with the column name on DataFrame"""
        self.type = kwargs.get('type', 'common')
        self.data_type = kwargs.get('data_type', None)
        """The type of the data column on Pandas Dataframe"""
        self.mask = kwargs.get('mask', None)
        """The mask for format the data column on Pandas Dataframe"""
        self.transformation = kwargs.get('transformation', None)
        """Pre processing transformation for the variable"""
        self.transformation_params = kwargs.get('transformation_params', None)
        self.partitioner = None
        """UoD partitioner for the variable data"""
        self.alpha_cut = kwargs.get('alpha_cut', 0.0)
        """Minimal membership value to be considered on fuzzyfication process"""


        if kwargs.get('data', None) is not None:
            self.build(**kwargs)

    def build(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        fs = kwargs.pop('partitioner', Grid.GridPartitioner)
        mf = kwargs.pop('func', Membership.trimf)
        np = kwargs.pop('npart', 10)
        data = kwargs.get('data', None)
        kw = kwargs.pop('partitioner_specific', {})
        self.partitioner = fs(data=data[self.data_label].values, npart=np, func=mf,
                              transformation=self.transformation, prefix=self.alias,
                              variable=self.name, **kw)

        self.partitioner.name = self.name + " " + self.partitioner.name

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
