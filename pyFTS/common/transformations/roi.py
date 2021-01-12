from pyFTS.common.transformations.transformation import Transformation 


class ROI(Transformation):
    """
    Return of Investment (ROI) transformation. Retrieved from Sadaei and Lee (2014) - Multilayer Stock
    Forecasting Model Using Fuzzy Time Series

    y'(t) = ( y(t) - y(t-1) ) / y(t-1)
    y(t) = ( y(t-1) * y'(t) ) + y(t-1)
    """
    def __init__(self, **kwargs):
        super(ROI, self).__init__()
        self.name = 'ROI'

    def apply(self, data, param=None, **kwargs):
        modified = [(data[i] - data[i - 1]) / data[i - 1] for i in np.arange(1, len(data))]
        modified.insert(0, .0)
        return modified

    def inverse(self, data, param=None, **kwargs):
        modified = [(param[i - 1] * data[i]) + param[i - 1] for i in np.arange(1, len(data))]
        return modified