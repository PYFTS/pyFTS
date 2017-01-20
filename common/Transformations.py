import numpy as np
import math
from pyFTS import *


def differential(original):
    n = len(original)
    diff = [original[t - 1] - original[t] for t in np.arange(1, n)]
    diff.insert(0, 0)
    return np.array(diff)


def boxcox(original, plambda):
    n = len(original)
    if plambda != 0:
        modified = [(original[t] ** plambda - 1) / plambda for t in np.arange(0, n)]
    else:
        modified = [math.log(original[t]) for t in np.arange(0, n)]
    return np.array(modified)


def Z(original):
    mu = np.mean(original)
    sigma = np.std(original)
    z = [(k - mu)/sigma for k in original]
    return z
