'''
Incremental Ensemble of FTS methods
'''


import numpy as np
import pandas as pd
from pyFTS.common import FuzzySet, FLR, fts, flrg
from pyFTS.models.ensemble import ensemble


class IncrementalEnsembleFTS(ensemble.EnsembleFTS):
    """
    Ensemble FTS
    """
    def __init__(self, **kwargs):
        super(IncrementalEnsembleFTS, self).__init__(**kwargs)
        self.shortname = "IncrementalEnsembleFTS"
        self.name = "Incremental Ensemble FTS"

        self.order = kwargs.get('order',1)

        self.order = kwargs.get('order', 1)

        self.partitioner_method = kwargs.get('partitioner_method', Grid.GridPartitioner)
        """The partitioner method to be called when a new model is build"""
        self.partitioner_params = kwargs.get('partitioner_params', {'npart': 10})
        """The partitioner method parameters"""
        self.partitioner = None
        """The most recent trained partitioner"""

        self.fts_method = kwargs.get('fts_method', None)
        """The FTS method to be called when a new model is build"""
        self.fts_params = kwargs.get('fts_params', {})
        """The FTS method specific parameters"""

        self.window_length = kwargs.get('window_length', 100)
        """The memory window length"""

        self.batch_size = kwargs.get('batch_size', 10)
        """The batch interval between each retraining"""
        self.is_high_order = True
        self.uod_clip = False
        self.max_lag = self.window_length + self.max_lag

    def train(self, data, **kwargs):

        self.partitioner = self.partitioner_method(data=data, **self.partitioner_params)
        self.model = self.fts_method(partitioner=self.partitioner, **self.fts_params)
        if self.model.is_high_order:
            self.model.order = self.model = self.fts_method(partitioner=self.partitioner,
                                                            order=self.order, **self.fts_params)
        self.model.fit(data, **kwargs)
        self.shortname = self.model.shortname






