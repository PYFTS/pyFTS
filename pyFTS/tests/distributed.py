from pyFTS.partitioners import Grid
from pyFTS.models import chen
from pyFTS.benchmarks import Measures
from pyFTS.common import Membership
from pyFTS.common import Util as cUtil, fts
import pandas as pd
import numpy as np
import os
from pyFTS.common import Transformations
from copy import deepcopy
from pyFTS.models import pwfts
from pyFTS.models.multivariate import common, variable, mvfts, wmvfts
from pyFTS.benchmarks import benchmarks as bchmk, Measures
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime

import time

from pyFTS.data import Malaysia, SONDA

sonda = SONDA.get_dataframe()[['datahora','glo_avg']].iloc[:600000]

sonda['data'] = pd.to_datetime(sonda["datahora"], format='%Y-%m-%d %H:%M:%S')

sonda = sonda.drop(sonda.index[np.where(sonda["glo_avg"] <= 0.01)])
sonda = sonda.dropna()

print(sonda)

sp = {'seasonality': DateTime.day_of_year , 'names': ['Jan','Fev','Mar','Abr','Mai','Jun','Jul', 'Ago','Set','Out','Nov','Dez']}

vmonth = variable.Variable("Month", data_label="data", partitioner=seasonal.TimeGridPartitioner, npart=12,
                           data=sonda, partitioner_specific=sp)

sp = {'seasonality': DateTime.minute_of_day, 'names': [str(k)+'hs' for k in range(0,24)]}

vhour = variable.Variable("Hour", data_label="data", partitioner=seasonal.TimeGridPartitioner, npart=24,
                          data=sonda, partitioner_specific=sp)

vavg = variable.Variable("Radiation", data_label="glo_avg", alias='rad',
                         partitioner=Grid.GridPartitioner, npart=35,
                         data=sonda)

model = wmvfts.WeightedMVFTS(explanatory_variables=[vhour, vhour, vavg], target_variable=vavg)

bchmk.distributed_model_train_test_time([model], sonda, 600000, 0.8, inc=1,
                                        num_batches=7, distributed='dispy',nodes=['192.168.0.106','192.168.0.110'],
                                        file='deho.db', tag='speedup', dataset='SONDA')


#model.fit(train_mv, num_batches=4, distributed='dispy',nodes=['192.168.0.106'])
#model.predict(test_mv, num_batches=4, distributed='dispy', nodes=['192.168.0.106'])

#print(model.__dict__['training_time'])
#print(model.__dict__['forecasting_time'])





'''
datasets = {}

sonda = SONDA.get_dataframe()[['datahora','glo_avg','ws_10m']]

sonda = sonda.drop(sonda.index[np.where(sonda["ws_10m"] <= 0.01)])
sonda = sonda.drop(sonda.index[np.where(sonda["glo_avg"] <= 0.01)])
sonda = sonda.dropna()

malaysia = Malaysia.get_dataframe()

datasets['SONDA.ws_10m'] = sonda["ws_10m"].values
datasets['SONDA.glo_avg'] = sonda["glo_avg"].values
datasets['Malaysia.temperature'] = malaysia["temperature"].values
datasets['Malaysia.load'] = malaysia["load"].values

windows = [600000, 600000, 10000, 10000]

for ct, (dataset_name, dataset) in enumerate(datasets.items()):
    bchmk.train_test_time(dataset, windowsize=windows[ct], train=0.9, inc=.5,
                     methods=[pwfts.ProbabilisticWeightedFTS],
                     order=2,
                     partitions=50,
                     steps=cpus,
                     num_batches=cpus,
                     distributed='dispy', nodes=['192.168.0.110', '192.168.0.107','192.168.0.106'],
                     file="experiments.db", dataset=dataset_name,
                     tag="speedup")
'''
