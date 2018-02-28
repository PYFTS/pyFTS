from pyFTS.common import fts


class Variable:
    def __init__(self,name, **kwargs):
        self.name = name
        self.alias = kwargs.get('alias', self.name)
        self.data_label = kwargs.get('alias', self.name)
        self.partitioner = kwargs.get('partitioner',None)
        self.type = kwargs.get('type', 'common')
        self.transformation = kwargs.get('transformation', None)

    def __str__(self):
        return self.name


class MVFTS(fts.FTS):
    def __init__(self, name, **kwargs):
        super(MVFTS, self).__init__(1, name, **kwargs)
        self.explanatory_variables = []
        self.target_variable = None

    def append_variable(self, var):
        self.explanatory_variables.append(var)


