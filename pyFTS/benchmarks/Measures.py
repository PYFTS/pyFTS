# -*- coding: utf8 -*-

"""
pyFTS module for common benchmark metrics
"""

import time
import numpy as np
import pandas as pd
from pyFTS.common import FuzzySet, SortedCollection
from pyFTS.probabilistic import ProbabilityDistribution


def acf(data, k):
    """
    Autocorrelation function estimative

    :param data: 
    :param k: 
    :return: 
    """
    mu = np.nanmean(data)
    sigma = np.var(data)
    n = len(data)
    s = 0
    for t in np.arange(0, n - k):
        s += (data[t] - mu) * (data[t + k] - mu)

    return 1 / ((n - k) * sigma) * s


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
    return np.nanmean(np.abs(np.divide(np.subtract(targets, forecasts), targets))) * 100


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
        return np.nanmean(np.abs(forecasts - targets) / ((forecasts + targets) / 2))
    elif type == 2:
        return np.nanmean(np.abs(forecasts - targets) / (np.abs(forecasts) + abs(targets))) * 100
    else:
        return np.sum(np.abs(forecasts - targets)) / np.sum(forecasts + targets)


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
    for k in np.arange(0, l - 1):
        y.append(np.subtract(forecasts[k], targets[k]) ** 2)
        naive.append(np.subtract(targets[k + 1], targets[k]) ** 2)
    return np.sqrt(np.divide(np.sum(y), np.sum(naive)))


def TheilsInequality(targets, forecasts):
    """
    Theil’s Inequality Coefficient

    :param targets: 
    :param forecasts: 
    :return: 
    """
    res = targets - forecasts
    t = len(res)
    us = np.sqrt(sum([u ** 2 for u in res]))
    ys = np.sqrt(sum([y ** 2 for y in targets]))
    fs = np.sqrt(sum([f ** 2 for f in forecasts]))
    return us / (ys + fs)


def sharpness(forecasts):
    """Sharpness - Mean size of the intervals"""
    tmp = [i[1] - i[0] for i in forecasts]
    return np.mean(tmp)


def resolution(forecasts):
    """Resolution - Standard deviation of the intervals"""
    shp = sharpness(forecasts)
    tmp = [abs((i[1] - i[0]) - shp) for i in forecasts]
    return np.mean(tmp)


def coverage(targets, forecasts):
    """Percent of target values that fall inside forecasted interval"""

    preds = []
    for i in np.arange(0, len(forecasts)):
        if targets[i] >= forecasts[i][0] and targets[i] <= forecasts[i][1]:
            preds.append(1)
        else:
            preds.append(0)
    return np.nanmean(preds)


def pinball(tau, target, forecast):
    """
    Pinball loss function. Measure the distance of forecast to the tau-quantile of the target

    :param tau: quantile value in the range (0,1)
    :param target: 
    :param forecast: 
    :return: float, distance of forecast to the tau-quantile of the target
    """

    if target >= forecast:
        return np.subtract(target, forecast) * tau
    else:
        return np.subtract(forecast, target) * (1 - tau)


def pinball_mean(tau, targets, forecasts):
    """
    Mean pinball loss value of the forecast for a given tau-quantile of the targets

    :param tau: quantile value in the range (0,1)
    :param targets: list of target values
    :param forecasts: list of prediction intervals
    :return: float, the pinball loss mean for tau quantile
    """

    if tau <= 0.5:
        preds = [pinball(tau, targets[i], forecasts[i][0]) for i in np.arange(0, len(forecasts))]
    else:
        preds = [pinball(tau, targets[i], forecasts[i][1]) for i in np.arange(0, len(forecasts))]
    return np.nanmean(preds)


def winkler_score(tau, target, forecast):
    """
    R. L. Winkler, A Decision-Theoretic Approach to Interval Estimation, J. Am. Stat. Assoc. 67 (337) (1972) 187–191. doi:10.2307/2284720.

    :param tau:
    :param target:
    :param forecast:
    :return:
    """

    delta = forecast[1] - forecast[0]
    if forecast[0] <= target <= forecast[1]:
        return delta
    elif forecast[0] > target:
        return delta + (2 * (forecast[0] - target)) / tau
    elif forecast[1] < target:
        return delta + (2 * (target - forecast[1])) / tau


def winkler_mean(tau, targets, forecasts):
    """
    Mean Winkler score value of the forecast for a given tau-quantile of the targets

    :param tau: quantile value in the range (0,1)
    :param targets: list of target values
    :param forecasts: list of prediction intervals
    :return: float, the Winkler score mean for tau quantile
    """

    preds = [winkler_score(tau, targets[i], forecasts[i]) for i in np.arange(0, len(forecasts))]

    return np.nanmean(preds)


def brier_score(targets, densities):
    """
    Brier Score for probabilistic forecasts.
    Brier (1950). "Verification of Forecasts Expressed in Terms of Probability". Monthly Weather Review. 78: 1–3.

    :param targets: a list with the target values
    :param densities: a list with pyFTS.probabil objectsistic.ProbabilityDistribution
    :return: float
    """

    if not isinstance(densities, list):
        densities = [densities]
        targets = [targets]

    ret = []

    for ct, d in enumerate(densities):
        try:
            v = d.bin_index.find_le(targets[ct])

            score = sum([d.density(k) ** 2 for k in d.bins if k != v])
            score += (d.density(v) - 1) ** 2
            ret.append(score)
        except ValueError as ex:
            ret.append(sum([d.density(k) ** 2 for k in d.bins]))
    return sum(ret) / len(ret)


def logarithm_score(targets, densities):
    """
    Logarithm Score for probabilistic forecasts.
    Good IJ (1952).  “Rational Decisions.”Journal of the Royal Statistical Society B,14(1),107–114. URLhttps://www.jstor.org/stable/2984087.

    :param targets: a list with the target values
    :param densities: a list with pyFTS.probabil objectsistic.ProbabilityDistribution
    :return: float
    """

    _ls = float(0.0)
    if isinstance(densities, ProbabilityDistribution.ProbabilityDistribution):
        densities = [densities]
        targets = [targets]

    n = len(densities)
    for ct, df in enumerate(densities):
        _ls += -np.log(df.density(targets[ct]))

    return _ls / n


def crps(targets, densities):
    """
    Continuous Ranked Probability Score

    :param targets: a list with the target values
    :param densities: a list with pyFTS.probabil objectsistic.ProbabilityDistribution
    :return: float
    """

    _crps = float(0.0)
    if isinstance(densities, ProbabilityDistribution.ProbabilityDistribution):
        densities = [densities]
        targets = [targets]

    n = len(densities)
    for ct, df in enumerate(densities):
        _crps += sum([(df.cumulative(bin) - (1 if bin >= targets[ct] else 0)) ** 2 for bin in df.bins])

    return _crps / n


def get_point_statistics(data, model, **kwargs):
    """
    Condensate all measures for point forecasters

    :param data: test data
    :param model: FTS model with point forecasting capability
    :param kwargs:
    :return: a list with the RMSE, SMAPE and U Statistic
    """

    steps_ahead = kwargs.get('steps_ahead', 1)
    kwargs['type'] = 'point'

    indexer = kwargs.get('indexer', None)

    if indexer is not None:
        ndata = np.array(indexer.get_data(data))
    elif model.is_multivariate:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Multivariate data must be a Pandas DataFrame!")
        ndata = data
    else:
        ndata = np.array(data)

    ret = list()

    if steps_ahead == 1:
        forecasts = model.predict(ndata, **kwargs)

        if model.is_multivariate and model.has_seasonality:
            ndata = model.indexer.get_data(ndata)
        elif model.is_multivariate:
            ndata = ndata[model.target_variable.data_label].values

        if not isinstance(forecasts, (list, np.ndarray)):
            forecasts = [forecasts]

        if len(forecasts) != len(ndata) - model.max_lag:
            forecasts = np.array(forecasts[:-1])
        else:
            forecasts = np.array(forecasts)

        ret.append(np.round(rmse(ndata[model.max_lag:], forecasts), 2))
        ret.append(np.round(mape(ndata[model.max_lag:], forecasts), 2))
        ret.append(np.round(UStatistic(ndata[model.max_lag:], forecasts), 2))
    else:
        steps_ahead_sampler = kwargs.get('steps_ahead_sampler', 1)
        nforecasts = []
        for k in np.arange(model.order, len(ndata) - steps_ahead, steps_ahead_sampler):
            sample = ndata[k - model.order: k]
            tmp = model.predict(sample, **kwargs)
            nforecasts.append(tmp[-1])

        start = model.max_lag + steps_ahead - 1
        ret.append(np.round(rmse(ndata[start:-1:steps_ahead_sampler], nforecasts), 2))
        ret.append(np.round(mape(ndata[start:-1:steps_ahead_sampler], nforecasts), 2))
        ret.append(np.round(UStatistic(ndata[start:-1:steps_ahead_sampler], nforecasts), 2))

    return ret


def get_interval_statistics(data, model, **kwargs):
    """
    Condensate all measures for point interval forecasters

    :param data: test data
    :param model: FTS model with interval forecasting capability
    :param kwargs:
    :return: a list with the sharpness, resolution, coverage, .05 pinball mean,
            .25 pinball mean, .75 pinball mean and .95 pinball mean.
    """

    steps_ahead = kwargs.get('steps_ahead', 1)
    kwargs['type'] = 'interval'

    ret = list()

    if steps_ahead == 1:
        forecasts = model.predict(data, **kwargs)
        ret.append(round(sharpness(forecasts), 2))
        ret.append(round(resolution(forecasts), 2))
        ret.append(round(coverage(data[model.max_lag:], forecasts[:-1]), 2))
        ret.append(round(pinball_mean(0.05, data[model.max_lag:], forecasts[:-1]), 2))
        ret.append(round(pinball_mean(0.25, data[model.max_lag:], forecasts[:-1]), 2))
        ret.append(round(pinball_mean(0.75, data[model.max_lag:], forecasts[:-1]), 2))
        ret.append(round(pinball_mean(0.95, data[model.max_lag:], forecasts[:-1]), 2))
        ret.append(round(winkler_mean(0.05, data[model.max_lag:], forecasts[:-1]), 2))
        ret.append(round(winkler_mean(0.25, data[model.max_lag:], forecasts[:-1]), 2))
    else:
        forecasts = []
        for k in np.arange(model.order, len(data) - steps_ahead):
            sample = data[k - model.order: k]
            tmp = model.predict(sample, **kwargs)
            forecasts.append(tmp[-1])

        start = model.max_lag + steps_ahead - 1
        ret.append(round(sharpness(forecasts), 2))
        ret.append(round(resolution(forecasts), 2))
        ret.append(round(coverage(data[model.max_lag:], forecasts), 2))
        ret.append(round(pinball_mean(0.05, data[start:], forecasts), 2))
        ret.append(round(pinball_mean(0.25, data[start:], forecasts), 2))
        ret.append(round(pinball_mean(0.75, data[start:], forecasts), 2))
        ret.append(round(pinball_mean(0.95, data[start:], forecasts), 2))
        ret.append(round(winkler_mean(0.05, data[start:], forecasts), 2))
        ret.append(round(winkler_mean(0.25, data[start:], forecasts), 2))
    return ret


def get_interval_ahead_statistics(data, intervals, **kwargs):
    """
    Condensate all measures for point interval forecasters

    :param data: test data
    :param model: FTS model with interval forecasting capability
    :param kwargs:
    :return: a list with the sharpness, resolution, coverage, .05 pinball mean,
            .25 pinball mean, .75 pinball mean and .95 pinball mean.
    """

    l = len(intervals)

    if len(data) != l:
        raise Exception("Data and intervals have different lenghts!")

    lags = {}

    for lag in range(l):
        ret = {}
        datum = data[lag]
        interval = intervals[lag]
        ret['steps'] = lag
        ret['method'] = ''
        ret['sharpness'] = round(interval[1] - interval[0], 2)
        ret['coverage'] = 1 if interval[0] <= datum <= interval[1] else 0
        ret['pinball05'] = round(pinball(0.05, datum, interval[0]), 2)
        ret['pinball25'] = round(pinball(0.25, datum, interval[0]), 2)
        ret['pinball75'] = round(pinball(0.75, datum, interval[1]), 2)
        ret['pinball95'] = round(pinball(0.95, datum, interval[1]), 2)
        ret['winkler05'] = round(winkler_score(0.05, datum, interval), 2)
        ret['winkler25'] = round(winkler_score(0.25, datum, interval), 2)
        lags[lag] = ret

    return lags


def get_distribution_statistics(data, model, **kwargs):
    """
    Get CRPS statistic and time for a forecasting model

    :param data: test data
    :param model: FTS model with probabilistic forecasting capability
    :param kwargs:
    :return: a list with the CRPS and execution time
    """

    steps_ahead = kwargs.get('steps_ahead', 1)
    kwargs['type'] = 'distribution'

    ret = list()

    if steps_ahead == 1:
        _s1 = time.time()
        forecasts = model.predict(data, **kwargs)
        _e1 = time.time()
        ret.append(round(crps(data[model.max_lag:], forecasts[:-1]), 3))
        ret.append(round(_e1 - _s1, 3))
        ret.append(round(brier_score(data[model.max_lag:], forecasts[:-1]), 3))
    else:
        skip = kwargs.get('steps_ahead_sampler', 1)
        forecasts = []
        _s1 = time.time()
        for k in np.arange(model.max_lag, len(data) - steps_ahead, skip):
            sample = data[k - model.max_lag: k]
            tmp = model.predict(sample, **kwargs)
            forecasts.append(tmp[-1])
        _e1 = time.time()

        start = model.max_lag + steps_ahead
        ret.append(round(crps(data[start:-1:skip], forecasts), 3))
        ret.append(round(_e1 - _s1, 3))
        ret.append(round(brier_score(data[start:-1:skip], forecasts), 3))
    return ret


def get_distribution_ahead_statistics(data, distributions):
    """
    Get CRPS statistic and time for a forecasting model

    :param data: test data
    :param model: FTS model with probabilistic forecasting capability
    :param kwargs:
    :return: a list with the CRPS and execution time
    """

    l = len(distributions)

    if len(data) != l:
        raise Exception("Data and distributions have different lenghts!")

    lags = {}

    for lag in range(l):
        ret = {}
        datum = data[lag]
        dist = distributions[lag]
        ret['steps'] = lag
        ret['method'] = ''
        ret['crps'] = round(crps(datum, dist), 3)
        ret['brier'] = round(brier_score(datum, dist), 3)
        ret['log'] = round(logarithm_score(datum, dist), 3)
        lags[lag] = ret

    return lags
