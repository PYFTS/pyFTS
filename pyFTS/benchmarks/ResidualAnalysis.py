#!/usr/bin/python
# -*- coding: utf8 -*-

"""Residual Analysis methods"""

import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
from pyFTS.common import Transformations,Util
from pyFTS.benchmarks import Measures
from scipy import stats


def residuals(targets, forecasts, order=1):
    """First order residuals"""
    return np.array(targets[order:]) - np.array(forecasts[:-1])


def chi_squared(q, h):
    """
    Chi-Squared value
    :param q: 
    :param h: 
    :return: 
    """
    p = stats.chi2.sf(q, h)
    return p


def compare_residuals(data, models):
    """
    Compare residual's statistics of several models
    :param data: 
    :param models: 
    :return: 
    """
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
        ret += str(chi_squared(q2, 10))
        ret += "	\\\\ \n"
    return ret


def plotResiduals(targets, models, tam=[8, 8], save=False, file=None):
    """
    Plot residuals and statistics
    :param targets: 
    :param models: 
    :param tam: 
    :param save: 
    :param file: 
    :return: 
    """
    fig, axes = plt.subplots(nrows=len(models), ncols=3, figsize=tam)
    for c, mfts in enumerate(models):
        if len(models) > 1:
            ax = axes[c]
        else:
            ax = axes
        forecasts = mfts.forecast(targets)
        res = residuals(targets,forecasts,mfts.order)
        mu = np.mean(res)
        sig = np.std(res)

        ax[0].set_title("Residuals Mean=" + str(mu) + " STD = " + str(sig))
        ax[0].set_ylabel('E')
        ax[0].set_xlabel('T')
        ax[0].plot(res)

        ax[1].set_title("Residuals Autocorrelation")
        ax[1].set_ylabel('ACS')
        ax[1].set_xlabel('Lag')
        ax[1].acorr(res)

        ax[2].set_title("Residuals Histogram")
        ax[2].set_ylabel('Freq')
        ax[2].set_xlabel('Bins')
        ax[2].hist(res)

        c += 1

    plt.tight_layout()

    Util.show_and_save_image(fig, file, save)


def plot_residuals(targets, models, tam=[8, 8], save=False, file=None):
    fig, axes = plt.subplots(nrows=len(models), ncols=3, figsize=tam)

    for c, mfts in enumerate(models, start=0):
        if len(models) > 1:
            ax = axes[c]
        else:
            ax = axes
        forecasts = mfts.forecast(targets)
        res = residuals(targets, forecasts, mfts.order)
        mu = np.mean(res)
        sig = np.std(res)

        if c == 0: ax[0].set_title("Residuals", size='large')
        ax[0].set_ylabel(mfts.shortname, size='large')
        ax[0].set_xlabel(' ')
        ax[0].plot(res)

        if c == 0: ax[1].set_title("Residuals Autocorrelation", size='large')
        ax[1].set_ylabel('ACS')
        ax[1].set_xlabel('Lag')
        ax[1].acorr(res)

        if c == 0: ax[2].set_title("Residuals Histogram", size='large')
        ax[2].set_ylabel('Freq')
        ax[2].set_xlabel('Bins')
        ax[2].hist(res)

    plt.tight_layout()

    Util.show_and_save_image(fig, file, save)


def single_plot_residuals(targets, forecasts, order, tam=[8, 8], save=False, file=None):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=tam)

    res = residuals(targets, forecasts, order)

    ax[0].set_title("Residuals", size='large')
    ax[0].set_ylabel("Model", size='large')
    ax[0].set_xlabel(' ')
    ax[0].plot(res)

    ax[1].set_title("Residuals Autocorrelation", size='large')
    ax[1].set_ylabel('ACS')
    ax[1].set_xlabel('Lag')
    ax[1].acorr(res)

    ax[2].set_title("Residuals Histogram", size='large')
    ax[2].set_ylabel('Freq')
    ax[2].set_xlabel('Bins')
    ax[2].hist(res)

    plt.tight_layout()

    Util.show_and_save_image(fig, file, save)
