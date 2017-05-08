"""
Kernel Density Estimation 
"""


class KernelSmoothing(object):
    """Kernel Density Estimation"""
    def __init__(self,h, data, method="epanechnikov"):
        self.h = h
        self.data = data
        self.method = method

    def kernel(self, u):
        if self.method == "epanechnikov":
            return (3/4) * (1 - u**2)
        elif self.method == "uniform":
            return 0.5
        elif self.method == "uniform":
            return 0.5

    def probability(self, x):
        l = len(self.data)
        p = sum([self.kernel((x - k)/self.h) for k in self.data]) / l*self.h

        return p