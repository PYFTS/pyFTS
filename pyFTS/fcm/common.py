from pyFTS.fcm import Activations
import numpy as np


class FuzzyCognitiveMap(object):
    def __init__(self, **kwargs):
        super(FuzzyCognitiveMap, self).__init__()
        self.order = kwargs.get('order',1)
        self.concepts = kwargs.get('partitioner',None)
        self.weights = []
        self.bias = []
        self.activation_function = kwargs.get('activation_function', Activations.sigmoid)

    def activate(self, concepts):
        dot_products = np.zeros(len(self.concepts))
        for k in np.arange(0, self.order):
            dot_products += np.dot(np.array(concepts[k]).T, self.weights[k]) + self.bias[k]
        return self.activation_function( dot_products )

