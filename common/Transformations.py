import numpy as np
import math
from pyFTS import *


def differential(original, lags=1):
    n = len(original)
    diff = [original[t - lags] - original[t] for t in np.arange(lags, n)]
    for t in np.arange(0, lags): diff.insert(0, 0)
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


# retrieved from Sadaei and Lee (2014) - Multilayer Stock ForecastingModel Using Fuzzy Time Series
def roi(original):
    n = len(original)
    roi = []
    for t in np.arange(0, n-1):
        roi.append( (original[t+1] - original[t])/original[t]  )
    return roi
