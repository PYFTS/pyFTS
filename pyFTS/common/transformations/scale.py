from pyFTS.common.transformations.transformation import Transformation 


class Scale(Transformation):
    """
    Scale data inside a interval [min, max]

    
    """
    def __init__(self, min=0, max=1):
        super(Scale, self).__init__()
        self.data_max = None
        self.data_min = None
        self.transf_max = max
        self.transf_min = min
        self.name = 'Scale'

    @property
    def parameters(self):
        return [self.transf_max, self.transf_min]

    def apply(self, data, param=None,**kwargs):
        if self.data_max is None:
            self.data_max = np.nanmax(data)
            self.data_min = np.nanmin(data)
        data_range = self.data_max - self.data_min
        transf_range = self.transf_max - self.transf_min
        if isinstance(data, list):
            tmp = [(k + (-1 * self.data_min)) / data_range for k in data]
            tmp2 = [ (k * transf_range) + self.transf_min for k in tmp]
        else:
            tmp = (data + (-1 * self.data_min)) / data_range
            tmp2 = (tmp * transf_range) + self.transf_min

        return  tmp2

    def inverse(self, data, param, **kwargs):
        data_range = self.data_max - self.data_min
        transf_range = self.transf_max - self.transf_min
        if isinstance(data, list):
            tmp2 = [(k - self.transf_min) / transf_range   for k in data]
            tmp = [(k * data_range) + self.data_min for k in tmp2]
        else:
            tmp2 = (data - self.transf_min) / transf_range
            tmp = (tmp2 * data_range) + self.data_min
        return tmp