from pyFTS.common.transformations.transformation import Transformation 


class BoxCox(Transformation):
    """
    Box-Cox power transformation

    y'(t) = log( y(t) )
    y(t) = exp( y'(t) )
    """
    def __init__(self, plambda):
        super(BoxCox, self).__init__()
        self.plambda = plambda
        self.name = 'BoxCox'

    @property
    def parameters(self):
        return self.plambda

    def apply(self, data, param=None, **kwargs):
        if self.plambda != 0:
            modified = [(dat ** self.plambda - 1) / self.plambda for dat in data]
        else:
            modified = [np.log(dat) for dat in data]
        return np.array(modified)

    def inverse(self, data, param=None, **kwargs):
        if self.plambda != 0:
            modified = [np.exp(np.log(dat * self.plambda + 1) ) / self.plambda for dat in data]
        else:
            modified = [np.exp(dat) for dat in data]
        return np.array(modified)