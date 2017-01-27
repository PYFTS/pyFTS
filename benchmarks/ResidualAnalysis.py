#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
from pyFTS.common import Transformations,Util
from pyFTS.benchmarks import Measures
from scipy import stats


def residuals(targets, forecasts, order=1):
    return np.array(targets[order:]) - np.array(forecasts[:-1])


def ChiSquared(q,h):
    p = stats.chi2.sf(q, h)
    return p


def compareResiduals(data, models):
    ret = "Model		& Order     & Mean      & STD       & Box-Pierce    & Box-Ljung & P-value \\\\ \n"
    for mfts in models:
        forecasts = mfts.forecast(data)
        res = residuals(data, forecasts, mfts.order)
        mu = np.mean(res)
        sig = np.std(res)
        ret += mfts.shortname + "		& "
        ret += str(mfts.order) + "		& "
        ret += str(round(mu,2)) + "		& "
        ret += str(round(sig,2)) + "		& "
        q1 = Measures.BoxPierceStatistic(res, 10)
        ret += str(round(q1,2)) + "		& "
        q2 = Measures.BoxLjungStatistic(res, 10)
        ret += str(round(q2,2)) + "		& "
        ret += str(ChiSquared(q2, 10))
        ret += "	\\\\ \n"
    return ret


def plotResiduals(targets, models, tam=[8, 8], save=False, file=None):

    fig, axes = plt.subplots(nrows=len(models), ncols=3, figsize=tam)
    c = 0
    for mfts in models:
        forecasts = mfts.forecast(targets)
        res = residuals(targets,forecasts,mfts.order)
        mu = np.mean(res)
        sig = np.std(res)

        axes[c][0].set_title("Residuals Mean=" + str(mu) + " STD = " + str(sig))
        axes[c][0].set_ylabel('E')
        axes[c][0].set_xlabel('T')
        axes[c][0].plot(res)

        axes[c][1].set_title("Residuals Autocorrelation")
        axes[c][1].set_ylabel('ACS')
        axes[c][1].set_xlabel('Lag')
        axes[c][1].acorr(res)

        axes[c][2].set_title("Residuals Histogram")
        axes[c][2].set_ylabel('Freq')
        axes[c][2].set_xlabel('Bins')
        axes[c][2].hist(res)

        c += 1

    plt.tight_layout()

    Util.showAndSaveImage(fig, file, save)


def plotResiduals2(targets, models, tam=[8, 8], save=False, file=None):
    fig, axes = plt.subplots(nrows=len(models), ncols=3, figsize=tam)

    for c, mfts in enumerate(models, start=0):
        forecasts = mfts.forecast(targets)
        res = residuals(targets, forecasts, mfts.order)
        mu = np.mean(res)
        sig = np.std(res)

        if c == 0: axes[c][0].set_title("Residuals", size='large')
        axes[c][0].set_ylabel(mfts.shortname, size='large')
        axes[c][0].set_xlabel(' ')
        axes[c][0].plot(res)

        if c == 0: axes[c][1].set_title("Residuals Autocorrelation", size='large')
        axes[c][1].set_ylabel('ACS')
        axes[c][1].set_xlabel('Lag')
        axes[c][1].acorr(res)

        if c == 0: axes[c][2].set_title("Residuals Histogram", size='large')
        axes[c][2].set_ylabel('Freq')
        axes[c][2].set_xlabel('Bins')
        axes[c][2].hist(res)

    plt.tight_layout()

    Util.showAndSaveImage(fig, file, save)
