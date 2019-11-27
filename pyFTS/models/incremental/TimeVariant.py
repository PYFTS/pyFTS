"""
Meta model that wraps another FTS method and continously retrain it using a data window with
the most recent data
"""

import numpy as np
from pyFTS.common import FuzzySet, FLR, fts, flrg
from pyFTS.partitioners import Grid


class Retrainer(fts.FTS):
    """
    Meta model for incremental/online learning that retrain its internal model after
    data windows controlled by the parameter 'batch_size', using as the training data a
    window of recent lags, whose size is controlled by the parameter 'window_length'.
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
        self.is_time_variant = True
        self.uod_clip = False
        self.max_lag = self.window_length + self.order
        self.is_wrapper = True

    def train(self, data, **kwargs):
        self.partitioner = self.partitioner_method(data=data, **self.partitioner_params)
        self.model = self.fts_method(partitioner=self.partitioner, **self.fts_params)
        if self.model.is_high_order:
            self.model.order = self.model = self.fts_method(partitioner=self.partitioner,
                                                            order=self.order, **self.fts_params)
        self.model.fit(data, **kwargs)
        self.shortname = "TimeVariant - " + self.model.shortname

    def forecast(self, data, **kwargs):
        l = len(data)

        no_update = kwargs.get('no_update',False)

        if no_update:
            return self.model.predict(data, **kwargs)

        horizon = self.window_length + self.order

        ret = []

        for k in np.arange(horizon, l+1):
            _train = data[k - horizon: k - self.order]
            _test = data[k - self.order: k]

            if k % self.batch_size == 0 or self.model is None:
                if self.auto_update:
                    self.model.train(_train)
                else:
                    self.train(_train, **kwargs)

            ret.extend(self.model.predict(_test, **kwargs))

        return ret

    def forecast_ahead(self, data, steps, **kwargs):
        if len(data) < self.order:
            return data

        if isinstance(data, np.ndarray):
            data = data.tolist()

        start = kwargs.get('start_at',0)

        ret = data[:start+self.order]
        for k in np.arange(start+self.order, steps+start+self.order):
            tmp = self.forecast(ret[k-self.order:k], no_update=True, **kwargs)

            if isinstance(tmp,(list, np.ndarray)):
                tmp = tmp[-1]

            ret.append(tmp)
            data.append(tmp)

        return ret[-steps:]

    def offset(self):
        return self.max_lag

    def __str__(self):
        """String representation of the model"""

        return str(self.model)

    def __len__(self):
        """
        The length (number of rules) of the model

        :return: number of rules
        """
        return len(self.model)
