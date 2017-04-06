# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
from pyFTS.common import FuzzySet,SortedCollection


# Autocorrelation function estimative
def acf(data, k):
    mu = np.mean(data)
    sigma = np.var(data)
    n = len(data)
    s = 0
    for t in np.arange(0,n-k):
        s += (data[t]-mu) * (data[t+k] - mu)

    return 1/((n-k)*sigma)*s


# Erro quadrático médio
def rmse(targets, forecasts):
    return np.sqrt(np.nanmean((targets - forecasts) ** 2))


def rmse_interval(targets, forecasts):
    fmean = [np.mean(i) for i in forecasts]
    return np.sqrt(np.nanmean((fmean - targets) ** 2))


# Erro Percentual médio
def mape(targets, forecasts):
    return np.mean(np.abs(targets - forecasts) / targets) * 100


def smape(targets, forecasts, type=2):
    if type == 1:
        return np.mean(np.abs(forecasts - targets) / ((forecasts + targets)/2))
    elif type == 2:
        return np.mean(np.abs(forecasts - targets) / (abs(forecasts) + abs(targets)) )*100
    else:
        return sum(np.abs(forecasts - targets)) / sum(forecasts + targets)


def mape_interval(targets, forecasts):
    fmean = [np.mean(i) for i in forecasts]
    return np.mean(abs(fmean - targets) / fmean) * 100


# Theil's U Statistic
def UStatistic(targets, forecasts):
    l = len(targets)
    naive = []
    y = []
    for k in np.arange(0,l-1):
        y.append((forecasts[k ] - targets[k ]) ** 2)
        naive.append((targets[k + 1] - targets[k]) ** 2)
    return np.sqrt(sum(y) / sum(naive))


# Theil’s Inequality Coefficient
def TheilsInequality(targets, forecasts):
    res = targets - forecasts
    t = len(res)
    us = np.sqrt(sum([u**2 for u in res]))
    ys = np.sqrt(sum([y**2 for y in targets]))
    fs = np.sqrt(sum([f**2 for f in forecasts]))
    return  us / (ys + fs)


# Q Statistic for Box-Pierce test
def BoxPierceStatistic(data, h):
    n = len(data)
    s = 0
    for k in np.arange(1,h+1):
        r = acf(data, k)
        s += r**2
    return n*s


# Q Statistic for Ljung–Box test
def BoxLjungStatistic(data, h):
    n = len(data)
    s = 0
    for k in np.arange(1,h+1):
        r = acf(data, k)
        s += r**2 / (n -k)
    return n*(n-2)*s


# Sharpness - Mean size of the intervals
def sharpness(forecasts):
    tmp = [i[1] - i[0] for i in forecasts]
    return np.mean(tmp)


# Resolution - Standard deviation of the intervals
def resolution(forecasts):
    shp = sharpness(forecasts)
    tmp = [abs((i[1] - i[0]) - shp) for i in forecasts]
    return np.mean(tmp)


# Percent of
def coverage(targets, forecasts):
    preds = []
    for i in np.arange(0, len(forecasts)):
        if targets[i] >= forecasts[i][0] and targets[i] <= forecasts[i][1]:
            preds.append(1)
        else:
            preds.append(0)
    return np.mean(preds)


def pmf_to_cdf(density):
    ret = []
    for row in density.index:
        tmp = []
        prev = 0
        for col in density.columns:
            prev += density[col][row] if not np.isnan(density[col][row]) else 0
            tmp.append( prev )
        ret.append(tmp)
    df = pd.DataFrame(ret, columns=density.columns)
    return df


def heavyside_cdf(bins, targets):
    ret = []
    for t in targets:
        result = [1 if b >= t else 0 for b in bins]
        ret.append(result)
    df = pd.DataFrame(ret, columns=bins)
    return df


# Continuous Ranked Probability Score
def crps(targets, densities):
    l = len(densities.columns)
    n = len(densities.index)
    Ff = pmf_to_cdf(densities)
    Fa = heavyside_cdf(densities.columns, targets)

    _crps = float(0.0)
    for k in densities.index:
        _crps += sum([ (Ff[col][k]-Fa[col][k])**2 for col in densities.columns])

    return _crps / float(l * n)


def get_point_statistics(data, model, indexer=None):
    if indexer is not None:
        ndata = np.array(indexer.get_data(data[model.order:]))
    else:
        ndata = np.array(data[model.order:])

    if model.isMultivariate or indexer is None:
        forecasts = model.forecast(data)
    elif not model.isMultivariate and indexer is not None:
        forecasts = model.forecast(indexer.get_data(data))

    if model.hasSeasonality:
        nforecasts = np.array(forecasts)
    else:
        nforecasts = np.array(forecasts[:-1])
    ret = list()
    try:
        ret.append(np.round(rmse(ndata, nforecasts), 2))
    except:
        ret.append(np.nan)
    try:
        ret.append(np.round(smape(ndata, nforecasts), 2))
    except:
        ret.append(np.nan)
    try:
        ret.append(np.round(UStatistic(ndata, nforecasts), 2))
    except:
        ret.append(np.nan)

    return ret


def get_interval_statistics(original, model):
    ret = list()
    forecasts = model.forecastInterval(original)
    ret.append(round(sharpness(forecasts), 2))
    ret.append(round(resolution(forecasts), 2))
    ret.append(round(coverage(original[model.order:], forecasts[:-1]), 2))
    return ret
