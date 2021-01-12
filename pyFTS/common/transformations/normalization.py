from pyFTS.common.transformations.transformation import Transformation 


class Normalization(Transformation):
    def __init__(self, **kwargs):
        super(Normalization, self).__init__()
        self.name = 'Normalization'

        self.mu = 0
        self.sigma = 0
        

    def train(self, data, **kwargs):
        self.mu = np.mean(data)
        self.sigma = np.std(data)

    def apply(self, data, param=None, **kwargs):
        modified = (data - self.mu) / self.sigma
        return modified

    def inverse(self, data, param=None, **kwargs):
        modified = (data * self.sigma) + self.mu
        return modified