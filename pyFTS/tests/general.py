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

dataset = pd.read_csv('https://query.data.world/s/2bgegjggydd3venttp3zlosh3wpjqj', sep=';')

dataset['data'] = pd.to_datetime(dataset["data"], format='%Y-%m-%d %H:%M:%S')

train_mv = dataset.iloc[:24505]
test_mv = dataset.iloc[24505:]

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

for ct, method in enumerate([mvfts.MVFTS, wmvfts.WeightedMVFTS,
                             cmvfts.ClusteredMVFTS, cmvfts.ClusteredMVFTS, cmvfts.ClusteredMVFTS]):

    if method != cmvfts.ClusteredMVFTS:
        model = method(explanatory_variables=[vmonth, vhour, vavg], target_variable=vavg, **parameters[ct])
    else:
        fs = grid.GridCluster(explanatory_variables=[vmonth, vhour, vavg], target_variable=vavg)
        model = method(explanatory_variables=[vmonth, vhour, vavg], target_variable=vavg, partitioner=fs,
                       **parameters[ct])

    model.shortname += str(ct)
    model.fit(train_mv)

    forecasts = model.predict(test_mv.iloc[:100])

    print(model.shortname, forecasts)



