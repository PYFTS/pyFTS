import numpy as np
import pandas as pd
import time

from pyFTS.data import Enrollments, TAIEX, SONDA
from pyFTS.partitioners import Grid, Simple, Entropy
from pyFTS.common import Util

from pyspark import SparkConf
from pyspark import SparkContext

import os
# make sure pyspark tells workers to use python3 not 2 if both are installed
os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/python3'
#'''


from pyFTS.models.multivariate import common, variable, wmvfts
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime
from pyFTS.partitioners import Grid

import matplotlib.pyplot as plt

'''
#fig, ax = plt.subplots(nrows=3, ncols=1, figsize=[15,5])


sp = {'seasonality': DateTime.day_of_year , 'names': ['Jan','Feb','Mar','Apr','May','Jun','Jul', 'Aug','Sep','Oct','Nov','Dec']}

vmonth = variable.Variable("Month", data_label="datahora", partitioner=seasonal.TimeGridPartitioner, npart=12, alpha_cut=.25,
                           data=train, partitioner_specific=sp)

#vmonth.partitioner.plot(ax[0])

sp = {'seasonality': DateTime.minute_of_day, 'names': [str(k) for k in range(0,24)]}

vhour = variable.Variable("Hour", data_label="datahora", partitioner=seasonal.TimeGridPartitioner, npart=24, alpha_cut=.2,
                          data=train, partitioner_specific=sp)

#vhour.partitioner.plot(ax[1])


vavg = variable.Variable("Radiation", data_label="glo_avg", alias='R',
                         partitioner=Grid.GridPartitioner, npart=35, alpha_cut=.3,
                         data=train)

#vavg.partitioner.plot(ax[2])

#plt.tight_layout()

#Util.show_and_save_image(fig, 'variables', True)

model = wmvfts.WeightedMVFTS(explanatory_variables=[vmonth,vhour,vavg], target_variable=vavg)


_s1 = time.time()
model.fit(train)
#model.fit(data, distributed='spark', url='spark://192.168.0.106:7077', num_batches=4)
_s2 = time.time()

print(_s2-_s1)

Util.persist_obj(model, 'sonda_wmvfts')
'''

#model = Util.load_obj('sonda_wmvfts')

'''
from pyFTS.benchmarks import Measures

_s1 = time.time()
print(Measures.get_point_statistics(test, model))
_s2 = time.time()

print(_s2-_s1)
'''

#print(len(model))


#

#model.fit(data, distributed='dispy', nodes=['192.168.0.110'])
#'''

'''
from pyFTS.models.multivariate import common, variable, mvfts, wmvfts, cmvfts, grid
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime

dataset = pd.read_csv('/home/petronio/Downloads/gefcom12.csv')
dataset = dataset.dropna()

train_mv = dataset.iloc[:15000]
test_mv = dataset.iloc[15000:]

from pyFTS.models.multivariate import common, variable, mvfts
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime


sp = {'seasonality': DateTime.minute_of_day, 'names': [str(k) for k in range(0,24)]}

vhour = variable.Variable("Hour", data_label="date", partitioner=seasonal.TimeGridPartitioner, npart=24,
                          data=train_mv, partitioner_specific=sp)

sp = {'seasonality': DateTime.day_of_week, 'names': ['mon','tue','wed','tur','fri','sat','sun']}

vday = variable.Variable("DayOfWeek", data_label="date", partitioner=seasonal.TimeGridPartitioner, npart=7,
                          data=train_mv, partitioner_specific=sp)

#sp = {'seasonality': DateTime.day_of_month, 'names': ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']}

sp = {'seasonality': DateTime.quarter}

vmonth = variable.Variable("Month", data_label="date", partitioner=seasonal.TimeGridPartitioner, npart=4,
                          data=train_mv, partitioner_specific=sp)


vload = variable.Variable("Load", data_label="load", alias='load',
                         partitioner=Grid.GridPartitioner, npart=20,
                         data=train_mv)

vtemp = variable.Variable("Temperature", data_label="temperature", alias='temperature',
                         partitioner=Grid.GridPartitioner, npart=20,
                         data=train_mv)

from pyFTS.models.multivariate import mvfts, wmvfts, cmvfts, grid

from pyFTS.models.multivariate import mvfts, wmvfts, cmvfts, grid

mtemp = wmvfts.WeightedMVFTS(explanatory_variables=[vhour, vmonth, vtemp], target_variable=vtemp)
mtemp.fit(train_mv)

Util.persist_obj(mtemp, 'mtemp')

from pyFTS.models import hofts

#mtemp = hofts.WeightedHighOrderFTS(order=2, partitioner=vtemp.partitioner)
#mtemp.fit(train_mv['temperature'].values)

from pyFTS.models.multivariate import mvfts, wmvfts, cmvfts, grid

mload = wmvfts.WeightedMVFTS(explanatory_variables=[vtemp, vload], target_variable=vload)
mload.fit(train_mv)

Util.persist_obj(mload, 'mload')


time_generator = lambda x : pd.to_datetime(x) + pd.to_timedelta(1, unit='h')


forecasts = mload.predict(test_mv.iloc[:1], steps_ahead=48, generators={'date': time_generator,
                                                                        'temperature': mtemp})

'''


data = pd.read_csv('https://query.data.world/s/6xfb5useuotbbgpsnm5b2l3wzhvw2i', sep=';')

train = data.iloc[:9000]
test = data.iloc[9000:9200]

from pyFTS.models.multivariate import common, variable, mvfts
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime
from pyFTS.partitioners import Grid

sp = {'seasonality': DateTime.day_of_year , 'names': ['Jan','Fev','Mar','Abr','Mai','Jun','Jul', 'Ago','Set','Out','Nov','Dez']}

vmonth = variable.Variable("Month", data_label="data", partitioner=seasonal.TimeGridPartitioner, npart=12,
                           data=train, partitioner_specific=sp, alpha_cut=.5)

sp = {'seasonality': DateTime.minute_of_day, 'names': [str(k) for k in range(0,24)]}

vhour = variable.Variable("Hour", data_label="data", partitioner=seasonal.TimeGridPartitioner, npart=24,
                          data=train, partitioner_specific=sp, alpha_cut=.5)

#print(vhour.partitioner)

#print(vmonth.partitioner.fuzzyfy(180))

vavg = variable.Variable("Radiation", data_label="glo_avg", alias='rad',
                         partitioner=Grid.GridPartitioner, npart=25, alpha_cut=.3,
                         data=train)

from pyFTS.models.multivariate import mvfts, wmvfts, cmvfts, grid

fs = grid.GridCluster(explanatory_variables=[vmonth, vhour, vavg], target_variable=vavg)


model = cmvfts.ClusteredMVFTS(explanatory_variables=[vmonth, vhour, vavg], target_variable=vavg,
                              partitioner=fs, knn=3)

model.fit(train)

model.predict(test)
