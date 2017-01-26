# -*- coding: utf8 -*-

import numpy as np
import pandas as pd


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
    return np.sqrt(np.nanmean((forecasts - targets) ** 2))


def rmse_interval(targets, forecasts):
    fmean = [np.mean(i) for i in forecasts]
    return np.sqrt(np.nanmean((fmean - targets) ** 2))


# Erro Percentual médio
def mape(targets, forecasts):
    return np.mean(abs(forecasts - targets) / forecasts) * 100


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

