import pandas as pd
import matplotlib.pylab as plt
from pyFTS.data import TAIEX as tx
from pyFTS.common import Transformations

from pyFTS.benchmarks import Measures
from pyFTS.partitioners import Grid, Util as pUtil
from pyFTS.common import Transformations, Util
from pyFTS.models.multivariate import common, variable, mvfts, wmvfts
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime

bc = Transformations.BoxCox(0)
tdiff = Transformations.Differential(1)

from pyFTS.models.multivariate import common, variable, mvfts, cmvfts
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime


from pyFTS.data import Malaysia

dataset = Malaysia.get_dataframe()

dataset["time"] = pd.to_datetime(dataset["time"], format='%m/%d/%y %I:%M %p')


data = dataset['load'].values

train_split = 8760


train_mv = dataset.iloc[:train_split]
test_mv = dataset.iloc[train_split:]

sp = {'seasonality': DateTime.month , #'type': 'common',
      'names': ['Jan','Feb','Mar','Apr','May','Jun','Jul', 'Aug','Sep','Oct','Nov','Dec']}

vmonth = variable.Variable("Month", data_label="time", partitioner=seasonal.TimeGridPartitioner, npart=12,
                           data=train_mv, partitioner_specific=sp)

sp = {'seasonality': DateTime.day_of_week, #'type': 'common',
      'names': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']}

vday = variable.Variable("Weekday", data_label="time", partitioner=seasonal.TimeGridPartitioner, npart=7,
                          data=train_mv, partitioner_specific=sp)

sp = {'seasonality': DateTime.hour_of_day} #, 'type': 'common'}

vhour = variable.Variable("Hour", data_label="time", partitioner=seasonal.TimeGridPartitioner, npart=24,
                          data=train_mv, partitioner_specific=sp)

vload = variable.Variable("load", data_label="load", partitioner=Grid.GridPartitioner, npart=10,
                           data=train_mv)

"""
model = cmvfts.ClusteredMVFTS(order=2, knn=3, cluster_params={'optmize': True})
model.append_variable(vmonthp)
model.append_variable(vdayp)
model.append_variable(vhourp)
model.append_variable(vload)
model.target_variable = vload
model.fit(train_mv)

print(len(model.cluster.sets.keys()))

model.cluster.prune()

print(len(model.cluster.sets.keys()))

model.predict(test_mv)
"""

'''
from pyFTS.data import Malaysia

dataset = Malaysia.get_dataframe()

dataset["date"] = pd.to_datetime(dataset["time"], format='%m/%d/%y %I:%M %p')

train_mv = dataset.iloc[:10000]
test_mv = dataset.iloc[10000:]

sp = {'seasonality': DateTime.month , 'names': ['Jan','Feb','Mar','Apr','May','Jun','Jul', 'Aug','Sep','Oct','Nov','Dec']}

vmonth = variable.Variable("Month", data_label="date", partitioner=seasonal.TimeGridPartitioner, npart=12, 
                         data=train_mv, partitioner_specific=sp)

sp = {'seasonality': DateTime.day_of_week, 'names': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']}

vday = variable.Variable("Weekday", data_label="date", partitioner=seasonal.TimeGridPartitioner, npart=7, 
                        data=train_mv, partitioner_specific=sp)

sp = {'seasonality': DateTime.hour_of_day}

vhour = variable.Variable("Hour", data_label="date", partitioner=seasonal.TimeGridPartitioner, npart=24, 
                        data=train_mv, partitioner_specific=sp)

vload = variable.Variable("load", data_label="load", partitioner=Grid.GridPartitioner, npart=10, 
                       data=train_mv)

vtemperature = variable.Variable("temperature", data_label="temperature", partitioner=Grid.GridPartitioner, npart=10, 
                       data=train_mv)

"""
variables = {
    'month': vmonth,
    'day': vday,
    'hour': vhour,
    'temperature': vtemperature,
    'load': vload
}

var_list = [k for k in variables.keys()]

models = []

import itertools

for k in [itertools.combinations(var_list, r) for r in range(2,len(var_list))]:
    for x in k:
      model = mvfts.MVFTS()
      print(x)
      for w in x:
        model.append_variable(variables[w])
        model.shortname += ' ' + w
      model.target_variable = vload
      model.fit(mv_train)
      models.append(model)
"""

"""
dataset =  pd.read_csv('/home/petronio/Downloads/priceHong')
dataset['hour'] = dataset.index.values % 24

data = dataset['price'].values.flatten()

train_split = 24 * 800

# Multivariate time series

train_mv = dataset.iloc[:train_split]
test_mv = dataset.iloc[train_split:]

#model = Util.load_obj('/home/petronio/Downloads/ClusteredMVFTS4')



vhour = variable.Variable("Hour", data_label="hour", partitioner=seasonal.TimeGridPartitioner, npart=24,
                          data=dataset,
                          partitioner_specific={'seasonality': DateTime.hour_of_day, 'type': 'common'})
vprice = variable.Variable("Price", data_label="price", partitioner=Grid.GridPartitioner, npart=55,
                            data=train_mv)
model = cmvfts.ClusteredMVFTS(order=2, knn=3)
model.append_variable(vhour)
model.append_variable(vprice)
model.target_variable = vprice
model.fit(train_mv)

data = [[1, 1.0], [2, 2.0]]

df = pd.DataFrame(data, columns=['hour','price'])

forecasts = model.predict(df, steps_ahead=24, generators={'Hour': lambda x : (x+1)%24 })
"""
'''

params = [
    {},
    {},
    {'order': 2, 'knn': 3, 'cluster_params': {'optmize': True}},
    {'order': 2, 'knn': 2, 'cluster_params': {'optmize': True}},
    {'order': 2, 'knn': 1, 'cluster_params': {'optmize': True}}
]

from pyFTS.models.multivariate import grid

cluster = None


for ct, method in enumerate([mvfts.MVFTS, wmvfts.WeightedMVFTS, cmvfts.ClusteredMVFTS, cmvfts.ClusteredMVFTS, cmvfts.ClusteredMVFTS]):

    model = method(**params[ct])
    model.append_variable(vmonth)
    model.append_variable(vday)
    model.append_variable(vhour)
    model.append_variable(vload)
    model.target_variable = vload
    model.fit(train_mv)

    if method == cmvfts.ClusteredMVFTS:
        model.cluster.prune()

    try:

        print(model.shortname, params[ct], Measures.get_point_statistics(test_mv, model))

    except Exception as ex:
        print(model.shortname, params[ct])
        print(ex)
        print("\n\n==============================================\n\n")

#print(model1)

#print(model1.predict(test_mv, steps_ahead=24, generators={'Hour': lambda x : (x+1)%24 }))

#'''