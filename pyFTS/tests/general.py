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
from pyFTS.models import chen, yu, cheng, ismailefendi, hofts, pwfts, tsaur, song, sadaei, ifts
from pyFTS.models.ensemble import ensemble
from pyFTS.common import Transformations, Membership, Util
from pyFTS.benchmarks import arima, quantreg, BSTS, gaussianproc, knn
from pyFTS.fcm import fts, common, GA

from pyFTS.data import TAIEX, NASDAQ, SP500

from pyFTS.data import TAIEX, NASDAQ, SP500
from pyFTS.common import Util

train = TAIEX.get_data()[1000:1800]
test = TAIEX.get_data()[1800:2000]

from pyFTS.models import pwfts
from pyFTS.partitioners import Grid

fs = Grid.GridPartitioner(data=train, npart=45)

model = pwfts.ProbabilisticWeightedFTS(partitioner=fs, order=1)
model.fit(train)
horizon = 10
'''
forecasts = model.predict(test[9:20], type='point')
intervals = model.predict(test[9:20], type='interval')
distributions = model.predict(test[9:20], type='distribution')

forecasts = model.predict(test[9:20], type='point', steps_ahead=horizon)
intervals = model.predict(test[9:20], type='interval', steps_ahead=horizon)
'''
distributions = model.predict(test[9:20], type='distribution', steps_ahead=horizon)


'''
train = TAIEX.get_data()[:800]
test = TAIEX.get_data()[800:1000]

order = 2
model = knn.KNearestNeighbors(order=order)
model.fit(train)

horizon=7

intervals05 = model.predict(test[:10], type='interval', alpha=.05, steps_ahead=horizon)

print(test[:10])
print(intervals05)

intervals25 = model.predict(test[:10], type='interval', alpha=.25, steps_ahead=horizon)
distributions = model.predict(test[:10], type='distribution', steps_ahead=horizon, smoothing=0.01, num_bins=100)

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=[15,5])
ax.plot(test[:10], label='Original',color='black')
Util.plot_interval2(intervals05, test[:10], start_at=model.order, ax=ax, color='green', label='alpha=.05'.format(model.order))
Util.plot_interval2(intervals25, test[:10], start_at=model.order, ax=ax, color='green', label='alpha=.25'.format(model.order))
Util.plot_distribution2(distributions, test[:10], start_at=model.order, ax=ax, cmap="Blues")

print("")
'''
