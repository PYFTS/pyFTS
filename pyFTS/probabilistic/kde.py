from pyFTS.common import Transformations
import numpy as np

"""
Kernel Density Estimation 
"""


class KernelSmoothing(object):
    """Kernel Density Estimation"""
    def __init__(self,h, method="epanechnikov"):
        self.h = h
        self.method = method
        self.transf = Transformations.Scale(min=0,max=1)

    def kernel(self, u):
        if self.method == "epanechnikov":
            return (3/4)*(1 - u**2)
        elif self.method == "gaussian":
            return (1/np.sqrt(2*np.pi))*np.exp(-0.5*u**2)
        elif self.method == "uniform":
            return 0.5

    def probability(self, x, data):
        l = len(data)

        ndata = self.transf.apply(data)
        nx = self.transf.apply(x)
        p = sum([self.kernel((nx - k)/self.h) for k in ndata]) / l*self.h

        return p