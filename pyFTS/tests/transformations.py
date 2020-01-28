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

'''
data = Enrollments.get_dataframe()

trend = Transformations.LinearTrend(data_field='Enrollments', index_field='Year')

trend.train(data)

plt.plot(data['Enrollments'].values)

plt.plot(trend.trend(data)) #)

detrend = trend.apply(data)

plt.plot(trend.inverse(detrend, data))
'''

#dataset = pd.read_csv('https://query.data.world/s/nxst4hzhjrqld4bxhbpn6twmjbwqk7')
#dataset['data'] = pd.to_datetime([str(y)+'-'+str(m) for y,m in zip(dataset['Ano'].values, dataset['Mes'].values)],
#                                  format='%Y-%m')
roi = Transformations.ROI()

'''

train = dataset['Total'].values[:30]
test = dataset['Total'].values[30:]

fs = Grid.GridPartitioner(data=train, npart=5, transformation=roi)

from pyFTS.models import hofts, pwfts

model = pwfts.ProbabilisticWeightedFTS(partitioner=fs, order=2)
#model = hofts.WeightedHighOrderFTS(partitioner=fs, order=1)
model.append_transformation(roi)


model.fit(train)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10,5])
ax.plot(test)

'''

'''
train = dataset.iloc[:30]
test = dataset.iloc[30:]

from pyFTS.models.multivariate import common, variable, mvfts, wmvfts, granular
from pyFTS.partitioners import Grid, Entropy
from pyFTS.models.seasonal.common import DateTime
from pyFTS.models.seasonal import partitioner as seasonal
sp = {'seasonality': DateTime.month , 'names': ['Jan','Fev','Mar','Abr','Mai','Jun','Jul', 'Ago','Set','Out','Nov','Dez']}

vmonth = variable.Variable("Month", data_label="data", partitioner=seasonal.TimeGridPartitioner, npart=12,
                           data=train, partitioner_specific=sp)

vtur = variable.Variable("Turistas", data_label="Total", alias='tur',
                         partitioner=Grid.GridPartitioner, npart=20, transformation=roi,
                         data=train)

#model = wmvfts.WeightedMVFTS(explanatory_variables=[vmonth, vtur], target_variable=vtur)
model = granular.GranularWMVFTS(explanatory_variables=[vmonth, vtur], target_variable=vtur, order=2, knn=1)
model.fit(train)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10,5])

ax.plot(test['Total'].values)

forecast = model.predict(test)

for k in np.arange(model.order):
  forecast.insert(0,None)

ax.plot(forecast)


plt.show()

print(dataset)
'''

eto = pd.read_csv('https://raw.githubusercontent.com/PatriciaLucas/Evapotranspiracao/master/ETo_setelagoas.csv', sep=',')
eto['Data'] = pd.to_datetime(eto["Data"], format='%Y-%m-%d')

from pyFTS.models.multivariate import common, variable, mvfts, wmvfts, granular
from pyFTS.models import hofts, pwfts
from pyFTS.partitioners import Grid, Entropy
from pyFTS.common import Membership
from pyFTS.models.seasonal.common import DateTime
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.benchmarks import Measures
from pyFTS.benchmarks import arima, quantreg, knn, benchmarks as bchmk

variables = {
    "Month": dict(data_label="Data", partitioner=seasonal.TimeGridPartitioner, npart=6),
    "Eto": dict(data_label="Eto", alias='eto',
                         partitioner=Grid.GridPartitioner, npart=50)
}

methods = [mvfts.MVFTS, wmvfts.WeightedMVFTS, granular.GranularWMVFTS]

time_generator = lambda x : pd.to_datetime(x) + pd.to_timedelta(1, unit='d')

parameters = [
    {},{},
    dict(fts_method=pwfts.ProbabilisticWeightedFTS, fuzzyfy_mode='both',
                order=1, knn=3)
]



bchmk.multivariate_sliding_window_benchmarks2(eto, 2000, train=0.8, inc=0.2,
                                              methods=methods,
                                              methods_parameters=parameters,
                                              variables=variables,
                                              target_variable='Eto',
                                              type='point',
                                              steps_ahead=[7],
                                              file="hyperparam.db", dataset='Eto',
                                              tag="experiments",
                                              generators= {'Data': time_generator}
                                              )

