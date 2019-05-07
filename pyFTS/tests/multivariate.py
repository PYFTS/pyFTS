import numpy as np
import pandas as pd
import time

from pyFTS.data import Enrollments, TAIEX, SONDA
from pyFTS.partitioners import Grid, Simple, Entropy
from pyFTS.common import Util

from pyFTS.models.multivariate import mvfts, wmvfts, cmvfts, grid, granular
from pyFTS.benchmarks import Measures
from pyFTS.common import Util as cUtil

from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime


from pyFTS.models.multivariate import common, variable, mvfts
from pyFTS.partitioners import Grid
from pyFTS.common import Membership


import os

'''
from pyFTS.data import lorentz
df = lorentz.get_dataframe(iterations=5000)

train = df.iloc[:4000]
#test = df.iloc[4000:]

npart=120


import sys


vx = variable.Variable("x", data_label="x", alias='x', partitioner=Grid.GridPartitioner,
                       partitioner_specific={'mf': Membership.gaussmf}, npart=npart, data=train)
vy = variable.Variable("y", data_label="y", alias='y', partitioner=Grid.GridPartitioner,
                       partitioner_specific={'mf': Membership.gaussmf}, npart=int(npart*1.5), data=train)
vz = variable.Variable("z", data_label="z", alias='z', partitioner=Grid.GridPartitioner,
                       partitioner_specific={'mf': Membership.gaussmf}, npart=int(npart*1.2), data=train)



rows = []

for ct, train, test in cUtil.sliding_window(df, windowsize=4100, train=.97, inc=.05):
    print('Window {}'.format(ct))
    for order in [1, 2, 3]:
        for knn in [1, 2, 3]:
            model = granular.GranularWMVFTS(explanatory_variables=[vx, vy, vz], target_variable=vx, order=order,
                                            knn=knn)

            model.fit(train)

            forecasts1 = model.predict(test, type='multivariate')
            forecasts2 = model.predict(test, type='multivariate', steps_ahead=100)

            for var in ['x', 'y', 'z']:
                row = [order, knn, var, len(model)]
                for horizon in [1, 25, 50, 75, 100]:
                    if horizon == 1:
                        row.append( Measures.mape(test[var].values[model.order:model.order+10],
                                             forecasts1[var].values[:10]))
                    else:
                        row.append( Measures.mape(test[var].values[:horizon],
                                                           forecasts2[var].values[:horizon]))

                print(row)
                rows.append(row)
                
columns = ['Order', 'knn', 'var', 'Rules']
for horizon in [1, 25, 50, 75, 100]:
    columns.append('h{}'.format(horizon))
final = pd.DataFrame(rows, columns=columns)

final.to_csv('gmvfts_lorentz1.csv',sep=';',index=False)
'''

import pandas as pd
df = pd.read_csv('https://query.data.world/s/ftb7bzgobr6bsg6bsuxuqowja6ew4r')

#df.dropna()

mload = np.nanmean(df["load"].values)
df['load'] = np.where(pd.isna(df["load"]), mload, df["load"])

mtemp = np.nanmean(df["temperature"].values)
df['temperature'] = np.where(pd.isna(df["temperature"]), mtemp, df["temperature"])

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')

df['hour'] = np.float64(df['date'].apply(lambda x: x.strftime('%H')))
df['weekday'] = np.float64(df['date'].apply(lambda x: x.strftime('%w')))
df['month'] = np.float64(df['date'].apply(lambda x: x.strftime('%m')))

train_mv = df.iloc[:31000]
test_mv = df.iloc[:31000:]

sp = {'seasonality': DateTime.minute_of_day, 'names': [str(k)+'hs' for k in range(0,24)]}

vhour = variable.Variable("Hour", data_label="date", partitioner=seasonal.TimeGridPartitioner, npart=24,
                          data=train_mv, partitioner_specific=sp, alpha_cut=.3)

vtemp = variable.Variable("Temperature", data_label="temperature", alias='temp',
                         partitioner=Grid.GridPartitioner, npart=15, func=Membership.gaussmf,
                         data=train_mv, alpha_cut=.3)

vload = variable.Variable("Load", data_label="load", alias='load',
                         partitioner=Grid.GridPartitioner, npart=20, func=Membership.gaussmf,
                         data=train_mv, alpha_cut=.3)

rows = []

time_generator = lambda x : pd.to_datetime(x) + pd.to_timedelta(1, unit='h')

for ct, train, test in cUtil.sliding_window(df, windowsize=32000, train=.98, inc=.05):
    print('Window {}'.format(ct))
    for order in [1, 2, 3]:
        for knn in [1, 2, 3]:
            model = granular.GranularWMVFTS(explanatory_variables=[vhour, vtemp, vload], target_variable=vload,
                                            order=order, knn=knn)

            model.fit(train)

            forecasts1 = model.predict(test, type='multivariate')
            forecasts2 = model.predict(test, type='multivariate', generators={'date': time_generator},
                                       steps_ahead=100)

            for var in ['temperature','load']:
                row = [order, knn, var, len(model)]
                for horizon in [1, 25, 50, 75, 100]:
                    if horizon == 1:
                        row.append(Measures.mape(test[var].values[model.order:model.order + 10],
                                                 forecasts1[var].values[:10]))
                    else:
                        row.append(Measures.mape(test[var].values[:horizon],
                                                 forecasts2[var].values[:horizon]))

                print(row)
                rows.append(row)

columns = ['Order', 'knn', 'var', 'Rules']
for horizon in [1, 25, 50, 75, 100]:
    columns.append('h{}'.format(horizon))
final = pd.DataFrame(rows, columns=columns)

final.to_csv('gmvfts_gefcom12.csv', sep=';', index=False)