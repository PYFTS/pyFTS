'''
Time Variant/Incremental Ensemble of FTS methods
'''


import numpy as np
import pandas as pd
from pyFTS.common import FuzzySet, FLR, fts, flrg
from pyFTS.partitioners import Grid
from pyFTS.models import hofts
from pyFTS.models.ensemble import ensemble


class IncrementalEnsembleFTS(ensemble.EnsembleFTS):
    """
    Time Variant/Incremental Ensemble of FTS methods
    """
    def __init__(self, **kwargs):
        super(IncrementalEnsembleFTS, self).__init__(**kwargs)
        self.shortname = "IncrementalEnsembleFTS"
        self.name = "Incremental Ensemble FTS"

        self.order = kwargs.get('order',1)

        self.partitioner_method = kwargs.get('partitioner_method', Grid.GridPartitioner)
        """The partitioner method to be called when a new model is build"""
        self.partitioner_params = kwargs.get('partitioner_params', {'npart': 10})
        """The partitioner method parameters"""

        self.fts_method = kwargs.get('fts_method', hofts.WeightedHighOrderFTS)
        """The FTS method to be called when a new model is build"""
        self.fts_params = kwargs.get('fts_params', {})
        """The FTS method specific parameters"""

        self.window_length = kwargs.get('window_length', 100)
        """The memory window length"""

        self.batch_size = kwargs.get('batch_size', 10)
        """The batch interval between each retraining"""

        self.num_models = kwargs.get('num_models', 5)
        """The number of models to hold in the ensemble"""

        self.point_method = kwargs.get('point_method', 'exponential')

        self.is_high_order = True
        self.uod_clip = False
        self.max_lag = self.window_length + self.order

    def train(self, data, **kwargs):

        partitioner = self.partitioner_method(data=data, **self.partitioner_params)
        model = self.fts_method(partitioner=partitioner, **self.fts_params)
        if model.is_high_order:
            model = self.fts_method(partitioner=partitioner, order=self.order, **self.fts_params)
        model.fit(data, **kwargs)
        self.append_model(model)
        if len(self.models) > self.num_models:
            self.models.pop(0)

    def forecast(self, data, **kwargs):
        l = len(data)

        data_window = []

        ret = []

        for k in np.arange(self.max_lag, l):

            k2 = k - self.max_lag

            data_window.append(data[k2])

            if k2 >= self.window_length:
                data_window.pop(0)

            if k % self.batch_size == 0 and k2 >= self.window_length:
                self.train(data_window, **kwargs)

            if len(self.models) > 0:
                sample = data[k2: k]
                tmp = self.get_models_forecasts(sample)
                point = self.get_point(tmp)
                ret.append(point)

        return ret






