from pyFTS.common import fts
from pyFTS.models import hofts
from pyFTS.fcm import common, GA, Activations, GD
import numpy as np


class FCM_FTS(hofts.HighOrderFTS):

    def __init__(self, **kwargs):
        super(FCM_FTS, self).__init__(**kwargs)
        self.fcm = common.FuzzyCognitiveMap(**kwargs)

    def train(self, data, **kwargs):
        '''
        GA.parameters['num_concepts'] = self.partitioner.partitions
        GA.parameters['order'] = self.order
        GA.parameters['partitioner'] = self.partitioner
        ret = GA.execute(data, **kwargs)
        self.fcm.weights = ret['weights']
        '''
        self.fcm.weights = GD.GD(data, self, alpha=0.01)

    def forecast(self, ndata, **kwargs):
        ret = []

        midpoints = np.array([fset.centroid for fset in self.partitioner])

        for t in np.arange(self.order, len(ndata)+1):

            sample = ndata[t - self.order : t]

            fuzzyfied = self.partitioner.fuzzyfy(sample, mode='vector')

            activation = self.fcm.activate(fuzzyfied)

            final = np.dot(midpoints, activation)/np.sum(activation)

            ret.append(final)

        return ret
