#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import numpy as np

import pandas as pd
from pyFTS.partitioners import Grid
from pyFTS.common import Transformations
from pyFTS.models import chen, hofts
from pyFTS.models.incremental import IncrementalEnsemble, TimeVariant

from pyFTS.data import AirPassengers, artificial

from pyFTS.models.ensemble import ensemble
from pyFTS.models import hofts
from pyFTS.data import TAIEX

data = TAIEX.get_data()

model = ensemble.EnsembleFTS()

for k in [15, 25, 35]:
    for order in [1, 2]:
        fs = Grid.GridPartitioner(data=data, npart=k)
        tmp = hofts.WeightedHighOrderFTS(partitioner=fs)

        tmp.fit(data)

        model.append_model(tmp)

forecasts = model.predict(data, type='interval', method='quantile', alpha=.05)

from pyFTS.benchmarks import benchmarks as bchmk

#f, ax = plt.subplots(1, 1, figsize=[20, 5])

#ax.plot(data)
#bchmk.plot_interval(ax, forecasts, 3, "")
print(forecasts)

'''
mu_local = 5
sigma_local = 0.25
mu_drift = 10
sigma_drift = 1.
deflen = 100
totlen = deflen * 10
order = 10


signal = artificial.SignalEmulator()\
.stationary_gaussian(mu_local,sigma_local,length=deflen//2,it=10)\
.stationary_gaussian(mu_drift,sigma_drift,length=deflen//2,it=10, additive=False)\
.run()

model2 = IncrementalEnsemble.IncrementalEnsembleFTS(partitioner_method=Grid.GridPartitioner, partitioner_params={'npart': 15},
                              fts_method=hofts.WeightedHighOrderFTS, fts_params={}, order=2 ,
                              batch_size=20, window_length=100, num_models=5)

forecasts = model2.predict(signal)

print(forecasts)
'''
'''
passengers = np.array(passengers["Passengers"])

e = ensemble.AllMethodEnsembleFTS(alpha=0.25, point_method="median", interval_method='quantile')

fo_methods = [song.ConventionalFTS, chen.ConventionalFTS, yu.WeightedFTS, cheng.TrendWeightedFTS, sadaei.ExponentialyWeightedFTS,
              ismailefendi.ImprovedWeightedFTS]

ho_methods = [hofts.HighOrderFTS, hwang.HighOrderFTS]

fs = Grid.GridPartitioner(passengers, 10, transformation=diff)

e.append_transformation(diff)

e.train(passengers, fs.sets, order=3)

"""

for method in fo_methods:
    model = method("")
    model.append_transformation(diff)
    model.train(passengers, fs.sets)
    e.append_model(model)


for method in ho_methods:
    for order in [1,2,3]:
        model = method("")
        model.append_transformation(diff)
        model.train(passengers, fs.sets, order=order)
        e.append_model(model)


arima100 = arima.ARIMA("", alpha=0.25)
#tmp.append_transformation(diff)
arima100.train(passengers, None, order=(1,0,0))

arima101 = arima.ARIMA("", alpha=0.25)
#tmp.append_transformation(diff)
arima101.train(passengers, None, order=(1,0,1))

arima200 = arima.ARIMA("", alpha=0.25)
#tmp.append_transformation(diff)
arima200.train(passengers, None, order=(2,0,0))

arima201 = arima.ARIMA("", alpha=0.25)
#tmp.append_transformation(diff)
arima201.train(passengers, None, order=(2,0,1))


e.append_model(arima100)
e.append_model(arima101)
e.append_model(arima200)
e.append_model(arima201)

e.train(passengers, None)


_mean = e.forecast(passengers, method="mean")
print(_mean)

_median = e.forecast(passengers, method="median")
print(_median)
"""
"""
_extremum = e.forecast_interval(passengers, method="extremum")
print(_extremum)

_quantile = e.forecast_interval(passengers, method="quantile", alpha=0.25)
print(_quantile)


_normal = e.forecast_interval(passengers, method="normal", alpha=0.25)
print(_normal)
"""

#"""
_extremum = e.forecast_ahead_interval(passengers, 10, method="extremum")
print(_extremum)

_quantile = e.forecast_ahead_interval(passengers[:50], 10, method="quantile", alpha=0.05)
print(_quantile)

_quantile = e.forecast_ahead_interval(passengers[:50], 10, method="quantile", alpha=0.25)
print(_quantile)

_normal = e.forecast_ahead_interval(passengers[:50], 10, method="normal", alpha=0.05)
print(_normal)
_normal = e.forecast_ahead_interval(passengers[:50], 10, method="normal", alpha=0.25)
print(_normal)
#"""

#dist = e.forecast_ahead_distribution(passengers, 20)

#print(dist)

#bchmk.plot_compared_intervals_ahead(passengers[:120],[e], ['blue','red'],
#                                    distributions=[True,False],  save=True, file="pictures/distribution_ahead_arma",
#                                    time_from=60, time_to=10, tam=[12,5])

'''


