import pandas as pd
import matplotlib.pylab as plt
from pyFTS.data import TAIEX, Malaysia
from pyFTS.common import Transformations

from pyFTS.benchmarks import Measures
from pyFTS.partitioners import Grid, Util as pUtil
from pyFTS.common import Transformations, Util
from pyFTS.models import pwfts
from pyFTS.models.multivariate import common, variable, mvfts, wmvfts
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime

bc = Transformations.BoxCox(0)
tdiff = Transformations.Differential(1)

from pyFTS.models.multivariate import common, variable, mvfts, cmvfts
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime

dataset = pd.read_csv('https://query.data.world/s/2bgegjggydd3venttp3zlosh3wpjqj', sep=';')

dataset['data'] = pd.to_datetime(dataset["data"], format='%Y-%m-%d %H:%M:%S')

data = dataset['glo_avg'].values

train_mv = dataset.iloc[:24505]
test_mv = dataset.iloc[24505:]



'''
model = Util.load_obj('/home/petronio/Downloads/ClusteredMVFTS1solarorder2knn3')

data = [[12, 100], [13, 200]]

for k in data:
    k[0] = pd.to_datetime('2018-01-01 {}:00:00'.format(k[0]), format='%Y-%m-%d %H:%M:%S')

df = pd.DataFrame(data, columns=['data', 'glo_avg'])

#forecasts = model.predict(df, steps_ahead=24, generators={'Hour': lambda x: x + pd.to_timedelta(1, unit='h')})

#print(forecasts)

f = lambda x: x + pd.to_timedelta(1, unit='h')

for ix, row in df.iterrows():
    print(row['data'])
    print(f(row['data']))
'''

# Multivariate time series
'''
dataset = pd.read_csv('https://query.data.world/s/2bgegjggydd3venttp3zlosh3wpjqj', sep=';')

dataset['data'] = pd.to_datetime(dataset["data"], format='%Y-%m-%d %H:%M:%S')

train_mv = dataset.iloc[:24505]
test_mv = dataset.iloc[24505:]

sp = {'seasonality': DateTime.minute_of_day, 'names': [str(k) for k in range(0,24)]}

vhour = variable.Variable("Hour", data_label="data", partitioner=seasonal.TimeGridPartitioner, npart=24,
                          data=train_mv, partitioner_specific=sp)

vavg = variable.Variable("Radiation", data_label="glo_avg", alias='rad',
                         partitioner=Grid.GridPartitioner, npart=30, alpha_cut=.3,
                         data=train_mv)

model = cmvfts.ClusteredMVFTS(pre_fuzzyfy=False, knn=3, fts_method=pwfts.ProbabilisticWeightedFTS)
model.append_variable(vhour)
model.append_variable(vavg)
model.target_variable = vavg
model.fit(train_mv)

Util.persist_obj(model, model.shortname)
'''

#model = Util.load_obj("ClusteredMVFTS")

model = Util.load_obj("ClusteredMVFTS2loadorder2knn2")

print(model)

print(model.predict(test_mv))

'''
train_mv = {}
test_mv = {}

models = {}

for key in ['price', 'solar', 'load']:
  models[key] = []

dataset =  pd.read_csv('/home/petronio/Downloads/priceHong')
dataset['hour'] = dataset.index.values % 24

data = dataset['price'].values.flatten()

train_split = 24 * 800


train_mv['price'] = dataset.iloc[:train_split]
test_mv['price'] = dataset.iloc[train_split:]

dataset = pd.read_csv('https://query.data.world/s/2bgegjggydd3venttp3zlosh3wpjqj', sep=';')

dataset['data'] = pd.to_datetime(dataset["data"], format='%Y-%m-%d %H:%M:%S')

train_mv['solar'] = dataset.iloc[:24505]
test_mv['solar'] = dataset.iloc[24505:]

from pyFTS.data import Malaysia

dataset = Malaysia.get_dataframe()

dataset["time"] = pd.to_datetime(dataset["time"], format='%m/%d/%y %I:%M %p')

train_mv['load'] = dataset.iloc[:train_split]
test_mv['load'] = dataset.iloc[train_split:]


exogenous = {}
endogenous = {}

for key in models.keys():
  exogenous[key] = {}

vhour = variable.Variable("Hour", data_label="hour", partitioner=seasonal.TimeGridPartitioner, npart=24,
                          data=train_mv['price'],
                          partitioner_specific={'seasonality': DateTime.hour_of_day, 'type': 'common'})
exogenous['price']['Hour'] = vhour

vprice = variable.Variable("Price", data_label="price", partitioner=Grid.GridPartitioner, npart=55,
                            data=train_mv['price'])
endogenous['price'] = vprice



sp = {'seasonality': DateTime.minute_of_day, 'names': [str(k) for k in range(0,24)]}

vhour = variable.Variable("Hour", data_label="data", partitioner=seasonal.TimeGridPartitioner, npart=24,
                          data=train_mv['solar'], partitioner_specific=sp)

exogenous['solar']['Hour'] = vhour

vavg = variable.Variable("Radiation", data_label="glo_avg", alias='rad',
                         partitioner=Grid.GridPartitioner, npart=30, alpha_cut=.3,
                         data=train_mv['solar'])

endogenous['solar'] = vavg


sp = {'seasonality': DateTime.hour_of_day}

vhourp = variable.Variable("Hour", data_label="time", partitioner=seasonal.TimeGridPartitioner, npart=24,
                          data=train_mv['load'], partitioner_specific=sp)

exogenous['load']['Hour'] = vhourp

vload = variable.Variable("load", data_label="load", partitioner=Grid.GridPartitioner, npart=10,
                           data=train_mv['load'])

endogenous['load'] = vload


from pyFTS.models.multivariate import mvfts, wmvfts, cmvfts

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=[15,15])

parameters = [
    {},{},
    {'order':2, 'knn': 1},
    {'order':2, 'knn': 2},
    {'order':2, 'knn': 3},
]

for ct, key in enumerate(models.keys()):

  for ct2, method in enumerate([mvfts.MVFTS, wmvfts.WeightedMVFTS,
                               cmvfts.ClusteredMVFTS,cmvfts.ClusteredMVFTS,cmvfts.ClusteredMVFTS]):
      print(key, method, parameters[ct2])
      model = method(**parameters[ct2])
      _key2 = ""
      for k in parameters[ct2].keys():
        _key2 += k + str(parameters[ct2][k])
      model.shortname += str(ct) + key + _key2
      for var in exogenous[key].values():
        model.append_variable(var)
      model.append_variable(endogenous[key])
      model.target_variable = endogenous[key]
      model.fit(train_mv[key])

      models[key].append(model.shortname)

      Util.persist_obj(model, model.shortname)

      del(model)
'''