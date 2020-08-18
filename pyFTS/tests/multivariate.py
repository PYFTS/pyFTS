import numpy as np
import pandas as pd
import time

from pyFTS.data import Enrollments, TAIEX, SONDA
from pyFTS.partitioners import Grid, Simple, Entropy
from pyFTS.common import Util

from pyFTS.models.multivariate import mvfts, wmvfts, cmvfts, grid, granular
from pyFTS.benchmarks import benchmarks as bchmk, Measures
from pyFTS.common import Util as cUtil
from pyFTS.models import pwfts
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime

from pyFTS.models.multivariate import common, variable, mvfts
from pyFTS.partitioners import Grid
from pyFTS.common import Membership
import os

from pyFTS.data import Malaysia, Enrollments


# Esta função cria o grid de data. Dada uma índice e a quantidade de dias out of range e in range ele retorna uma lista com a janela móvel
def gen_dates(index, time_is, time_os):
    t = -1
    t_aux = t
    size = len(index)
    dates = []
    while -size < t - time_is - time_os + 1:
        t = t_aux
        end_os = index[t]
        t -= time_os - 1
        init_os = index[t]
        t -= 1
        t_aux = t
        end_is = index[t]
        t -= time_is - 1
        init_is = index[t]
        t -= 1
        row = [init_is, end_is, init_os, end_os]
        dates.append(row)
    return dates


sp500 = pd.read_csv('/home/petronio/Downloads/sp500.csv', index_col=0)
stock = sp500.iloc[:, :5]

date_grid = gen_dates (index = stock.index, time_is= 100, time_os = 2)

date_range = date_grid[0]
init_is, end_is, init_os, end_os = date_range
train = stock[init_is:end_is]
test = stock[init_os:end_os]


close = variable.Variable("close", data_label='Adj Close', partitioner=Grid.GridPartitioner, npart=20,data=train)
polarity = variable.Variable("polarity", data_label='sentiment_bert', partitioner=Grid.GridPartitioner, npart=50,data=train)

from pyFTS.models import hofts
#mpolarity = mvfts.MVFTS(explanatory_variables=[close, polarity], target_variable=polarity)
mpolarity = hofts.HighOrderFTS(partitioner=polarity.partitioner)
mpolarity.fit(train['sentiment_bert'].values)

mclose = mvfts.MVFTS(explanatory_variables=[close, polarity], target_variable=close)
mclose.fit(train)

forecasts = mclose.predict(train[-1:], steps_ahead=2, generators = {'sentiment_bert': mpolarity})

'''
df = Malaysia.get_dataframe()
df['time'] = pd.to_datetime(df["time"], format='%m/%d/%y %I:%M %p')

train_mv = df.iloc[:1800]
test_mv = df.iloc[1800:2000]

del(df)

sp = {'seasonality': DateTime.minute_of_day, 'names': [str(k)+'hs' for k in range(0,24)]}

vhour = variable.Variable("Hour", data_label="time", partitioner=seasonal.TimeGridPartitioner, npart=24,
                          data=train_mv, partitioner_specific=sp, alpha_cut=.3)

vtemp = variable.Variable("Temperature", data_label="temperature", alias='temp',
                         partitioner=Grid.GridPartitioner, npart=15, func=Membership.gaussmf,
                         data=train_mv, alpha_cut=.3)

vload = variable.Variable("Load", data_label="load", alias='load',
                         partitioner=Grid.GridPartitioner, npart=20, func=Membership.trimf,
                         data=train_mv, alpha_cut=.3)

model = mvfts.MVFTS(explanatory_variables=[vhour, vtemp, vload], target_variable=vload)
#fs = Grid.GridPartitioner(data=Enrollments.get_data(), npart=10)
#print(fs)
#model = pwfts.ProbabilisticWeightedFTS(partitioner=vload.partitioner, order=2)
model.fit(train_mv) #, num_batches=10) #, distributed='dispy',nodes=['192.168.0.110'])
#model.fit(Enrollments.get_data()) #, num_batches=20) #, distributed='dispy',nodes=['192.168.0.110'])

print(model)


def sample_by_hour(data):
    return [np.nanmean(data[k:k+60]) for k in np.arange(0,len(data),60)]

def sample_date_by_hour(data):
    return [data[k] for k in np.arange(0,len(data),60)]

from pyFTS.data import SONDA

sonda = SONDA.get_dataframe()[['datahora','glo_avg','ws_10m']]

sonda = sonda.drop(sonda.index[np.where(sonda["ws_10m"] <= 0.01)])
sonda = sonda.drop(sonda.index[np.where(sonda["glo_avg"] <= 0.01)])
sonda = sonda.dropna()
sonda['datahora'] = pd.to_datetime(sonda["datahora"], format='%Y-%m-%d %H:%M:%S')


var = {
    'datahora': sample_date_by_hour(sonda['datahora'].values),
    'glo_avg': sample_by_hour(sonda['glo_avg'].values),
    'ws_10m': sample_by_hour(sonda['ws_10m'].values)
}

df = pd.DataFrame(var)


from pyFTS.data import Malaysia

df = Malaysia.get_dataframe()
df['time'] = pd.to_datetime(df["time"], format='%m/%d/%y %I:%M %p')

sp = {'seasonality': DateTime.minute_of_day, 'names': [str(k)+'hs' for k in range(0,24)]}

variables = {
    "Hour": dict(data_label="time", partitioner=seasonal.TimeGridPartitioner, npart=24,
                          partitioner_specific=sp, alpha_cut=.3),
    "Temperature": dict(data_label="temperature", alias='temp',
                        partitioner=Grid.GridPartitioner, npart=10, func=Membership.gaussmf,
                        alpha_cut=.25),
    "Load": dict(data_label="load", alias='load',
                         partitioner=Grid.GridPartitioner, npart=10, func=Membership.gaussmf,
                         alpha_cut=.25)
}


methods = [granular.GranularWMVFTS]

parameters = [
    dict(fts_method=pwfts.ProbabilisticWeightedFTS, fuzzyfy_mode='both',
                order=1, knn=1)
]

bchmk.multivariate_sliding_window_benchmarks2(df, 10000, train=0.9, inc=0.25,
                                              methods=methods,
                                              methods_parameters=parameters,
                                              variables=variables,
                                              target_variable='Temperature',
                                              type='distribution',
                                              steps_ahead=[1],
                                              file="experiments.db", dataset='Malaysia.temperature',
                                              tag="experiments"
                                              )





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
'''
