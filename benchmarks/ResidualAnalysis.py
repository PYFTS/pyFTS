#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
from pyFTS.common import Transformations


def residuals(targets, forecasts, order=1):
    return np.array(targets[order:]) - np.array(forecasts[:-order])


def plotResiduals(targets, forecasts, order=1, tam=[8, 8]):
    res = residuals(targets,forecasts,order)
    fig = plt.figure(figsize=tam)
    ax1 = fig.add_axes([0, 1, 0.9, 0.3])  # left, bottom, width, height
    ax1.plot(res)
    ax2 = fig.add_axes([0, 0.65, 0.9, 0.3])
    ax2.acorr(res)
    ax3 = fig.add_axes([0, 0.3, 0.9, 0.3])
    ax3.hist(res)

