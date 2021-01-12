
class Transformation(object):
    """
    Data transformation used on pre and post processing of the FTS
    """

    def __init__(self, **kwargs):
        self.is_invertible = True
        self.is_multivariate = False
        """detemine if this transformation can be applied to multivariate data"""
        self.minimal_length = 1
        self.name = ''

    def apply(self, data, param, **kwargs):
        """
        Apply the transformation on input data

        :param data: input data
        :param param:
        :param kwargs:
        :return: numpy array with transformed data
        """
        pass

    def inverse(self,data, param, **kwargs):
        """

        :param data: transformed data
        :param param:
        :param kwargs:
        :return: numpy array with inverse transformed data
        """
        pass

    def __str__(self):
        return self.name