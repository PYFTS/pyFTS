
import numpy as np
from pyFTS.common import FuzzySet, FLR, fts, flrg
from pyFTS.models import hofts
from pyFTS.models.multivariate import mvfts, grid, common


class ClusteredMVFTS(mvfts.MVFTS):
    """
    Meta model for multivariate, high order, clustered multivariate FTS
    """
    def __init__(self, **kwargs):
        super(ClusteredMVFTS, self).__init__(**kwargs)

        self.cluster_method = kwargs.get('cluster_method', grid.GridCluster)
        """The cluster method to be called when a new model is build"""
        self.cluster_params = kwargs.get('cluster_params', {})
        """The cluster method parameters"""
        self.cluster = None
        """The most recent trained clusterer"""

        self.fts_method = kwargs.get('fts_method', hofts.WeightedHighOrderFTS)
        """The FTS method to be called when a new model is build"""
        self.fts_params = kwargs.get('fts_params', {})
        """The FTS method specific parameters"""
        self.model = None
        """The most recent trained model"""

        self.is_high_order = True

        self.order = kwargs.get("order", 2)
        self.lags = kwargs.get("lags", None)
        self.alpha_cut = kwargs.get('alpha_cut', 0.25)

    def fuzzyfy(self,data):
        ndata = []
        for ct in range(1, len(data.index)):
            ix = data.index[ct - 1]
            data_point = self.format_data(data.loc[ix])
            ndata.append(common.fuzzyfy_instance_clustered(data_point, self.cluster, self.alpha_cut))

        return ndata


    def train(self, data, **kwargs):

        self.cluster = self.cluster_method(data=data, mvfts=self)

        self.model = self.fts_method(partitioner=self.cluster, **self.fts_params)
        if self.model.is_high_order:
            self.model.order = self.model = self.fts_method(partitioner=self.cluster,
                                                            order=self.order, **self.fts_params)

        ndata = self.fuzzyfy(data)

        self.model.train(ndata, fuzzyfied=True)
        self.shortname = self.model.shortname


    def forecast(self, ndata, **kwargs):

        ndata = self.fuzzyfy(ndata)

        return self.model.forecast(ndata, fuzzyfied=True, **kwargs)



    def __str__(self):
        """String representation of the model"""

        tmp = self.model.shortname + ":\n"
        for r in self.model.flrgs:
            tmp = tmp + str(self.model.flrgs[r]) + "\n"
        return tmp

    def __len__(self):
        """
        The length (number of rules) of the model

        :return: number of rules
        """
        return len(self.model)

