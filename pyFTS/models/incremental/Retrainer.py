"""
Meta model that wraps another FTS method and continously retrain it using a data window with the most recent data
"""

import numpy as np
from pyFTS.common import FuzzySet, FLR, fts, flrg
from pyFTS.partitioners import Grid


class Retrainer(fts.FTS):
    """
    Meta model for incremental/online learning
    """
    def __init__(self, **kwargs):
        super(Retrainer, self).__init__(**kwargs)

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
        self.model = None
        """The most recent trained model"""

        self.window_length  = kwargs.get('window_length',100)
        """The memory window length"""
        self.auto_update = False
        """If true the model is updated at each time and not recreated"""
        self.batch_size = kwargs.get('batch_size', 10)
        """The batch interval between each retraining"""
        self.is_high_order = True
        self.uod_clip = False
        
    @property
    def shortname(self):
        if self.model is None:
            self.model = self.fts_method()
            
        return self.model.shortname

    def train(self, data, **kwargs):
        self.partitioner = self.partitioner_method(data=data, **self.partitioner_params)
        self.model = self.fts_method(partitioner=self.partitioner, order=self.order, **self.fts_params)
        self.model.fit(data, **kwargs)

    def forecast(self, data, **kwargs):
        l = len(data)

        horizon = self.window_length + self.order

        ret = []

        for k in np.arange(horizon, l):
            _train = data[k - horizon: k - self.order]
            _test = data[k - self.order: k]

            if k % self.batch_size == 0 or self.model is None:
                print("Treinando {}".format(k))
                if self.auto_update:
                    self.model.train(_train)
                else:
                    self.train(_train, **kwargs)

            ret.extend(self.model.predict(_test, **kwargs))

        return ret

    def __str__(self):
        """String representation of the model"""

        return str(self.model)

    def __len__(self):
        """
        The length (number of rules) of the model

        :return: number of rules
        """
        return len(self.model)
