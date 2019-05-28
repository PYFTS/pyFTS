# -*- coding: utf8 -*-

"""
Kernel Density Estimation 
"""

from pyFTS.common import Transformations
import numpy as np


class KernelSmoothing(object):
    """Kernel Density Estimation"""
    def __init__(self, **kwargs):
        self.h = kwargs.get('h',.5)
        """Width parameter"""
        self.kernel = kwargs.get("kernel", "epanechnikov")
        """Kernel function"""
        self.data = kwargs.get("data",None)
        self.transf = Transformations.Scale(min=0,max=1)

        if self.data is not None:
            self.data = self.transf.apply(self.data)

    def kernel_function(self, u):
        """
        Apply the kernel

        :param u:
        :return:
        """
        if self.kernel == "epanechnikov":
            tmp = (3/4)*(1.0 - u**2)
            return tmp if tmp > 0 else 0
        elif self.kernel == "gaussian":
            return (1.0/np.sqrt(2*np.pi))*np.exp(-0.5*u**2)
        elif self.kernel == "uniform":
            return 0.5
        elif self.kernel == "triangular":
            tmp = 1.0 - np.abs(u)
            return tmp if tmp > 0 else 0
        elif self.kernel == "logistic":
            return 1.0/(np.exp(u)+2+np.exp(-u))
        elif self.kernel == "cosine":
            return (np.pi/4.0)*np.cos((np.pi/2.0)*u)
        elif self.kernel == "sigmoid":
            return (2.0/np.pi)*(1.0/(np.exp(u)+np.exp(-u)))
        elif self.kernel == "tophat":
            return 1 if np.abs(u) < 0.5 else 0
        elif self.kernel == "exponential":
            return 0.5 * np.exp(-np.abs(u))

    def probability(self, x, **kwargs):
        """
        Probability of the point x on data

        :param x:
        :param data:
        :return:
        """

        if self.data is None:
            self.data = kwargs.get('data',None)
            self.data = self.transf.apply(self.data)

        l = len(self.data)

        nx = self.transf.apply(x)
        p = sum([self.kernel_function((nx - k)/self.h) for k in self.data]) / l*self.h

        return p