
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
        self.cluster = kwargs.get('cluster', None)
        """The trained clusterer"""

        self.fts_method = kwargs.get('fts_method', hofts.WeightedHighOrderFTS)
        """The FTS method to be called when a new model is build"""
        self.fts_params = kwargs.get('fts_params', {})
        """The FTS method specific parameters"""
        self.model = None
        """The most recent trained model"""
        self.knn = kwargs.get('knn', 2)

        self.is_high_order = True

        self.order = kwargs.get("order", 2)
        self.lags = kwargs.get("lags", None)
        self.alpha_cut = kwargs.get('alpha_cut', 0.25)

        self.shortname = "ClusteredMVFTS"
        self.name = "Clustered Multivariate FTS"

        self.pre_fuzzyfy = kwargs.get('pre_fuzzyfy', True)

    def fuzzyfy(self,data):
        ndata = []
        for index, row in data.iterrows():
            data_point = self.format_data(row)
            ndata.append(common.fuzzyfy_instance_clustered(data_point, self.cluster, self.alpha_cut))

        return ndata

    def train(self, data, **kwargs):

        if self.cluster is None:
            self.cluster = self.cluster_method(data=data, mvfts=self, neighbors=self.knn, **self.cluster_params)

        self.model = self.fts_method(partitioner=self.cluster, **self.fts_params)
        if self.model.is_high_order:
            self.model.order = self.order

        if self.pre_fuzzyfy:
            ndata = self.fuzzyfy(data)
        else:
            ndata = [self.format_data(k) for k in data.to_dict('records')]

        self.model.train(ndata, fuzzyfied=self.pre_fuzzyfy)

        self.cluster.prune()

    def forecast(self, ndata, **kwargs):

        if self.pre_fuzzyfy:
            ndata = self.fuzzyfy(ndata)
        else:
            ndata = [self.format_data(k) for k in ndata.to_dict('records')]

        return self.model.forecast(ndata, fuzzyfied=self.pre_fuzzyfy, **kwargs)

    def __str__(self):
        """String representation of the model"""
        return str(self.model)

    def __len__(self):
        """
        The length (number of rules) of the model

        :return: number of rules
        """
        return len(self.model)

