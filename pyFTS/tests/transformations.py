#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import numpy as np
import matplotlib.pylab as plt

import pandas as pd

from pyFTS.common import Util as cUtil, FuzzySet
from pyFTS.partitioners import Grid, Entropy, Util as pUtil, Simple
from pyFTS.benchmarks import benchmarks as bchmk, Measures
from pyFTS.models import chen, yu, cheng, ismailefendi, hofts, pwfts, tsaur, song, sadaei, ifts
from pyFTS.models.ensemble import ensemble
from pyFTS.common import Membership, Util
from pyFTS.benchmarks import arima, quantreg, BSTS, gaussianproc, knn
from pyFTS.common import Transformations

tdiff = Transformations.Differential(1)

boxcox = Transformations.BoxCox(0)

from pyFTS.data import Enrollments, AirPassengers

'''
data = AirPassengers.get_data()

roi = Transformations.ROI()

#plt.plot(data)

_roi = roi.apply(data)

#plt.plot(_roi)

plt.plot(roi.inverse(_roi, data))
'''

'''
data = AirPassengers.get_dataframe()
data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m')
trend = Transformations.LinearTrend(data_field='Passengers', index_field='Month',
                                    index_type='datetime', datetime_mask='%Y-%d')

trend.train(data)

plt.plot(data['Passengers'].values)

plt.plot(trend.trend(data))

detrend = trend.apply(data)

plt.plot(trend.inverse(detrend, data, date_offset=pd.DateOffset(months=1)))
'''


data = Enrollments.get_dataframe()

trend = Transformations.LinearTrend(data_field='Enrollments', index_field='Year')

trend.train(data)

plt.plot(data['Enrollments'].values)

plt.plot(trend.trend(data)) #)

detrend = trend.apply(data)

plt.plot(trend.inverse(detrend, data))


plt.show()

print(data)
