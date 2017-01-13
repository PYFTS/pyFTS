#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd

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
