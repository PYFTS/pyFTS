"""
Kernel Density Estimation 
"""


class KernelSmoothing(object):
    """Kernel Density Estimation"""
    def __init__(self,h, method="epanechnikov"):
        self.h = h
        self.method = method

    def kernel(self, u):
        if self.method == "epanechnikov":
            return (3/4) * (1 - u**2)
        elif self.method == "gaussian":
            return 0.5
        elif self.method == "uniform":
            return 0.5

    def probability(self, x, data):
        l = len(data)
        p = sum([self.kernel((x - k)/self.h) for k in data]) / l*self.h

        return p