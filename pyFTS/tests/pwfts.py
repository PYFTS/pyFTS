import os
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import importlib
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pyFTS.common import Util
from pyFTS.benchmarks import benchmarks as bchmk, Measures
from pyFTS.models import pwfts,hofts,ifts
from pyFTS.models.multivariate import granular, grid
from pyFTS.partitioners import Grid, Util as pUtil

from pyFTS.models.multivariate import common, variable, mvfts
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime
from pyFTS.common import Membership

from pyFTS.data import SONDA, Malaysia

df = Malaysia.get_dataframe()
df['time'] = pd.to_datetime(df["time"], format='%m/%d/%y %I:%M %p')

train_mv = df.iloc[:8000]
test_mv = df.iloc[8000:10000]

sp = {'seasonality': DateTime.minute_of_day, 'names': [str(k)+'hs' for k in range(0,24)]}

vhour = variable.Variable("Hour", data_label="time", partitioner=seasonal.TimeGridPartitioner, npart=24,
                          data=train_mv, partitioner_specific=sp, alpha_cut=.3)
vtemp = variable.Variable("Temperature", data_label="temperature", alias='temp',
                         partitioner=Grid.GridPartitioner, npart=5, func=Membership.gaussmf,
                         data=train_mv, alpha_cut=.3)
vload = variable.Variable("Load", data_label="load", alias='load',
                         partitioner=Grid.GridPartitioner, npart=5, func=Membership.gaussmf,
                         data=train_mv, alpha_cut=.3)

order = 1
knn = 1

model = granular.GranularWMVFTS(explanatory_variables=[vhour, vtemp, vload], target_variable=vload,
                                fts_method=pwfts.ProbabilisticWeightedFTS, fuzzyfy_mode='both',
                                order=order, knn=knn)

model.fit(train_mv)


temp_generator = pwfts.ProbabilisticWeightedFTS(partitioner=vtemp.partitioner, order=2)
temp_generator.fit(train_mv['temperature'].values)

#print(model)

time_generator = lambda x : pd.to_datetime(x) + pd.to_timedelta(1, unit='h')
#temp_generator = lambda x : x

generators = {'time': time_generator, 'temperature': temp_generator}

#print(model.predict(test_mv.iloc[:10], type='point', steps_ahead=10, generators=generators))
#print(model.predict(test_mv.iloc[:10], type='interval', steps_ahead=10, generators=generators))
print(model.predict(test_mv.iloc[:10], type='distribution', steps_ahead=10, generators=generators))


#

#forecasts1 = model.predict(test_mv, type='multivariate')
#forecasts2 = model.predict(test, type='multivariate', generators={'date': time_generator},
#                           steps_ahead=200)


'''
from pyFTS.data import Enrollments
train = Enrollments.get_data()

fs = Grid.GridPartitioner(data=train, npart=10) #, transformation=tdiff)

model = pwfts.ProbabilisticWeightedFTS(partitioner=fs, order=2)
model.fit(train)
print(model)

print(model.predict(train))
'''