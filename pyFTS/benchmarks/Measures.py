# -*- coding: utf8 -*-

"""
pyFTS module for common benchmark metrics
"""

import time
import numpy as np
import pandas as pd
from pyFTS.common import FuzzySet,SortedCollection
from pyFTS.probabilistic import ProbabilityDistribution


def acf(data, k):
    """
    Autocorrelation function estimative
    :param data: 
    :param k: 
    :return: 
    """
    mu = np.mean(data)
    sigma = np.var(data)
    n = len(data)
    s = 0
    for t in np.arange(0,n-k):
        s += (data[t]-mu) * (data[t+k] - mu)

    return 1/((n-k)*sigma)*s


def rmse(targets, forecasts):
    """
    Root Mean Squared Error
    :param targets: 
    :param forecasts: 
    :return: 
    """
    if isinstance(targets, list):
        targets = np.array(targets)
    if isinstance(forecasts, list):
        forecasts = np.array(forecasts)
    return np.sqrt(np.nanmean((targets - forecasts) ** 2))


def rmse_interval(targets, forecasts):
    """
    Root Mean Squared Error
    :param targets: 
    :param forecasts: 
    :return: 
    """
    fmean = [np.mean(i) for i in forecasts]
    return np.sqrt(np.nanmean((fmean - targets) ** 2))


def mape(targets, forecasts):
    """
    Mean Average Percentual Error
    :param targets: 
    :param forecasts: 
    :return: 
    """
    if isinstance(targets, list):
        targets = np.array(targets)
    if isinstance(forecasts, list):
        forecasts = np.array(forecasts)
    return np.mean(np.abs(targets - forecasts) / targets) * 100


def smape(targets, forecasts, type=2):
    """
    Symmetric Mean Average Percentual Error
    :param targets: 
    :param forecasts: 
    :param type: 
    :return: 
    """
    if isinstance(targets, list):
        targets = np.array(targets)
    if isinstance(forecasts, list):
        forecasts = np.array(forecasts)
    if type == 1:
        return np.mean(np.abs(forecasts - targets) / ((forecasts + targets)/2))
    elif type == 2:
        return np.mean(np.abs(forecasts - targets) / (abs(forecasts) + abs(targets)) )*100
    else:
        return sum(np.abs(forecasts - targets)) / sum(forecasts + targets)


def mape_interval(targets, forecasts):
    fmean = [np.mean(i) for i in forecasts]
    return np.mean(abs(fmean - targets) / fmean) * 100


def UStatistic(targets, forecasts):
    """
    Theil's U Statistic
    :param targets: 
    :param forecasts: 
    :return: 
    """
    l = len(targets)
    if isinstance(targets, list):
        targets = np.array(targets)
    if isinstance(forecasts, list):
        forecasts = np.array(forecasts)

    naive = []
    y = []
    for k in np.arange(0,l-1):
        y.append((forecasts[k ] - targets[k ]) ** 2)
        naive.append((targets[k + 1] - targets[k]) ** 2)
    return np.sqrt(sum(y) / sum(naive))


def TheilsInequality(targets, forecasts):
    """
    Theil’s Inequality Coefficient
    :param targets: 
    :param forecasts: 
    :return: 
    """
    res = targets - forecasts
    t = len(res)
    us = np.sqrt(sum([u**2 for u in res]))
    ys = np.sqrt(sum([y**2 for y in targets]))
    fs = np.sqrt(sum([f**2 for f in forecasts]))
    return  us / (ys + fs)


def BoxPierceStatistic(data, h):
    """
    Q Statistic for Box-Pierce test
    :param data: 
    :param h: 
    :return: 
    """
    n = len(data)
    s = 0
    for k in np.arange(1,h+1):
        r = acf(data, k)
        s += r**2
    return n*s


def BoxLjungStatistic(data, h):
    """
    Q Statistic for Ljung–Box test
    :param data: 
    :param h: 
    :return: 
    """
    n = len(data)
    s = 0
    for k in np.arange(1,h+1):
        r = acf(data, k)
        s += r**2 / (n -k)
    return n*(n-2)*s


def sharpness(forecasts):
    """Sharpness - Mean size of the intervals"""
    tmp = [i[1] - i[0] for i in forecasts]
    return np.mean(tmp)


def resolution(forecasts):
    """Resolution - Standard deviation of the intervals"""
    shp = sharpness(forecasts)
    tmp = [abs((i[1] - i[0]) - shp) for i   in forecasts]
    return np.mean(tmp)


def coverage(targets, forecasts):
    """Percent of target values that fall inside forecasted interval"""
    preds = []
    for i in np.arange(0, len(forecasts)):
        if targets[i] >= forecasts[i][0] and targets[i] <= forecasts[i][1]:
            preds.append(1)
        else:
            preds.append(0)
    return np.mean(preds)


def pinball(tau, target, forecast):
    """
    Pinball loss function. Measure the distance of forecast to the tau-quantile of the target 
    :param tau: quantile value in the range (0,1)
    :param target: 
    :param forecast: 
    :return: distance of forecast to the tau-quantile of the target
    """
    if target >= forecast:
        return (target - forecast) * tau
    else:
        return (forecast - target) * (1 - tau)


def pinball_mean(tau, targets, forecasts):
    """
    Mean pinball loss value of the forecast for a given tau-quantile of the targets
    :param tau: quantile value in the range (0,1)
    :param targets: list of target values
    :param forecasts: list of prediction intervals
    :return: 
    """
    preds = []
    if tau <= 0.5:
        preds = [pinball(tau, targets[i], forecasts[i][0]) for i in np.arange(0, len(forecasts))]
    else:
        preds = [pinball(tau, targets[i], forecasts[i][1]) for i in np.arange(0, len(forecasts))]
    return np.nanmean(preds)


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


def crps(targets, densities):
    """Continuous Ranked Probability Score"""
    _crps = float(0.0)
    if isinstance(densities, pd.DataFrame):
        l = len(densities.columns)
        n = len(densities.index)
        Ff = pmf_to_cdf(densities)
        Fa = heavyside_cdf(densities.columns, targets)
        for k in densities.index:
            _crps += sum([ (Ff[col][k]-Fa[col][k])**2 for col in densities.columns])
    elif isinstance(densities, ProbabilityDistribution.ProbabilityDistribution):
        l = len(densities.bins)
        n = 1
        Fa = heavyside_cdf(densities.bins, targets)
        _crps = sum([(densities.cummulative(val) - Fa[val][0]) ** 2 for val in densities.bins])
    elif isinstance(densities, list):
        l = len(densities[0].bins)
        n = len(densities)
        Fa = heavyside_cdf(densities[0].bins, targets)
        for df in densities:
            _crps += sum([(df.cummulative(val) - Fa[val][0]) ** 2 for val in df.bins])

    return _crps / float(l * n)


def get_point_statistics(data, model, indexer=None):
    """Condensate all measures for point forecasters"""

    if indexer is not None:
        ndata = np.array(indexer.get_data(data))
    else:
        ndata = np.array(data[model.order:])

    if model.is_multivariate or indexer is None:
        forecasts = model.forecast(data)
    elif not model.is_multivariate and indexer is not None:
        forecasts = model.forecast(indexer.get_data(data))

    try:
        if model.has_seasonality:
            nforecasts = np.array(forecasts)
        else:
            nforecasts = np.array(forecasts[:-1])
    except Exception as ex:
        print(ex)
        return [np.nan,np.nan,np.nan]
    ret = list()
    try:
        ret.append(np.round(rmse(ndata, nforecasts), 2))
    except Exception as ex:
        print('Error in RMSE: {}'.format(ex))
        ret.append(np.nan)
    try:
        ret.append(np.round(smape(ndata, nforecasts), 2))
    except Exception as ex:
        print('Error in SMAPE: {}'.format(ex))
        ret.append(np.nan)
    try:
        ret.append(np.round(UStatistic(ndata, nforecasts), 2))
    except Exception as ex:
        print('Error in U: {}'.format(ex))
        ret.append(np.nan)

    return ret


def get_interval_statistics(original, model):
    """Condensate all measures for point_to_interval forecasters"""
    ret = list()
    forecasts = model.forecast_interval(original)
    ret.append(round(sharpness(forecasts), 2))
    ret.append(round(resolution(forecasts), 2))
    ret.append(round(coverage(original[model.order:], forecasts[:-1]), 2))
    ret.append(round(pinball_mean(0.05, original[model.order:], forecasts[:-1]), 2))
    ret.append(round(pinball_mean(0.25, original[model.order:], forecasts[:-1]), 2))
    ret.append(round(pinball_mean(0.75, original[model.order:], forecasts[:-1]), 2))
    ret.append(round(pinball_mean(0.95, original[model.order:], forecasts[:-1]), 2))
    return ret


def get_distribution_statistics(original, model, steps, resolution):
    ret = list()
    try:
        _s1 = time.time()
        densities1 = model.forecast_ahead_distribution(original, steps, parameters=3)
        _e1 = time.time()
        ret.append(round(crps(original, densities1), 3))
        ret.append(round(_e1 - _s1, 3))
    except Exception as e:
        print('Erro: ', e)
        ret.append(np.nan)
        ret.append(np.nan)

    try:
        _s2 = time.time()
        densities2 = model.forecast_ahead_distribution(original, steps, parameters=2)
        _e2 = time.time()
        ret.append( round(crps(original, densities2), 3))
        ret.append(round(_e2 - _s2, 3))
    except:
        ret.append(np.nan)
        ret.append(np.nan)

    return ret
