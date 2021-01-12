from pyFTS.common.transformations.transformation import Transformation 

class AdaptiveExpectation(Transformation):
    """
    Adaptive Expectation post processing
    """
    def __init__(self, parameters):
        super(AdaptiveExpectation, self).__init__(parameters)
        self.h = parameters
        self.name = 'AdaptExpect'

    @property
    def parameters(self):
        return self.parameters

    def apply(self, data, param=None,**kwargs):
        return data

    def inverse(self, data, param,**kwargs):
        n = len(data)

        inc = [param[t] + self.h*(data[t] - param[t]) for t in np.arange(0, n)]

        if n == 1:
            return inc[0]
        else:
            return inc