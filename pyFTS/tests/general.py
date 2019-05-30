#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import numpy as np
import matplotlib.pylab as plt
#from mpl_toolkits.mplot3d import Axes3D

import pandas as pd

from pyFTS.common import Util as cUtil, FuzzySet
from pyFTS.partitioners import Grid, Entropy, Util as pUtil, Simple
from pyFTS.benchmarks import benchmarks as bchmk, Measures
from pyFTS.models import chen, yu, cheng, ismailefendi, hofts, pwfts, tsaur, song, sadaei
from pyFTS.common import Transformations, Membership
from pyFTS.benchmarks import arima, quantreg, BSTS
from pyFTS.fcm import fts, common, GA

from pyFTS.data import Enrollments, TAIEX

data = TAIEX.get_data()

train = data[:500]
test = data[500:1000]

model = quantreg.QuantileRegression(order=1, dist=True)
model.fit(train)

horizon=5

#points  = model.predict(test[:10], type='point', steps_ahead=horizon)

intervals = model.predict(test[:10], type='interval', alpha=.05, smoothing=0.01, steps_ahead=horizon)
print(test[:10])
print(intervals)
distributions = model.predict(test[:10], type='distribution', steps_ahead=horizon, num_bins=100)


fig, ax = plt.subplots(nrows=1, ncols=1,figsize=[15,5])

ax.plot(test[:10], label='Original',color='black')
cUtil.plot_interval(ax, intervals, model.order, label='ensemble')
cUtil.plot_distribution2(distributions, test[:10], start_at=2, ax=ax, cmap="Blues")

'''
model = fts.FCM_FTS(partitioner=fs, order=1)

model.fcm.weights = np.array([
    [1, 1, 0, -1, -1],
    [1, 1, 1, 0, -1],
    [0, 1, 1, 1, 0],
    [-1, 0, 1, 1, 1],
    [-1, -1, 0, 1, 1]
])

print(data)
print(model.forecast(data))
'''
'''
dataset = pd.read_csv('https://query.data.world/s/2bgegjggydd3venttp3zlosh3wpjqj', sep=';')

dataset['data'] = pd.to_datetime(dataset["data"], format='%Y-%m-%d %H:%M:%S')

train_mv = dataset.iloc[:24505]
test_mv = dataset.iloc[24505:24605]

from itertools import product

levels = ['VL', 'L', 'M', 'H', 'VH']
sublevels = [str(k) for k in np.arange(0, 7)]
names = []
for combination in product(*[levels, sublevels]):
    names.append(combination[0] + combination[1])

print(names)

from pyFTS.models.multivariate import common, variable, mvfts
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime



sp = {'seasonality': DateTime.day_of_year , 'names': ['Jan','Feb','Mar','Apr','May',
                                                      'Jun','Jul', 'Aug','Sep','Oct',
                                                      'Nov','Dec']}

vmonth = variable.Variable("Month", data_label="data", partitioner=seasonal.TimeGridPartitioner, npart=12,
                           data=train_mv, partitioner_specific=sp)



sp = {'seasonality': DateTime.minute_of_day, 'names': [str(k)+'hs' for k in range(0,24)]}

vhour = variable.Variable("Hour", data_label="data", partitioner=seasonal.TimeGridPartitioner, npart=24,
                          data=train_mv, partitioner_specific=sp)


vavg = variable.Variable("Radiation", data_label="glo_avg", alias='rad',
                         partitioner=Grid.GridPartitioner, npart=35, partitioner_specific={'names': names},
                         data=train_mv)

from pyFTS.models.multivariate import mvfts, wmvfts, cmvfts, grid

parameters = [
    {}, {},
    {'order': 2, 'knn': 1},
    {'order': 2, 'knn': 2},
    {'order': 2, 'knn': 3},
]

from pyFTS.models.multivariate import mvfts, wmvfts, cmvfts, grid, granular
from pyFTS.benchmarks import Measures

time_generator = lambda x : pd.to_datetime(x) + pd.to_timedelta(1, unit='h')

model = granular.GranularWMVFTS(explanatory_variables=[vmonth, vhour, vavg], target_variable=vavg, order=2, knn=2)

model.fit(train_mv)

forecasts = model.predict(test_mv, type='multivariate', generators={'data': time_generator}, steps_ahead=24 )

print(forecasts)

'''
'''
from pyFTS.data import lorentz
df = lorentz.get_dataframe(iterations=5000)

train = df.iloc[:4000]
test = df.iloc[4000:]

from pyFTS.models.multivariate import common, variable, mvfts
from pyFTS.partitioners import Grid

vx = variable.Variable("x", data_label="x", alias='x', partitioner=Grid.GridPartitioner, npart=45, data=train)
vy = variable.Variable("y", data_label="y", alias='y', partitioner=Grid.GridPartitioner, npart=45, data=train)
vz = variable.Variable("z", data_label="z", alias='z', partitioner=Grid.GridPartitioner, npart=45, data=train)

from pyFTS.models.multivariate import mvfts, wmvfts, cmvfts, grid, granular
from pyFTS.benchmarks import Measures

model = granular.GranularWMVFTS(explanatory_variables=[vx, vy, vz], target_variable=vx, order=5, knn=2)

model.fit(train)

forecasts = model.predict(test, type='multivariate', steps_ahead=20)

print(forecasts)
'''