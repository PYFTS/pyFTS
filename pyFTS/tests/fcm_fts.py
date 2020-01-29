from pyFTS.fcm import Activations
import numpy as np
import os
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pyFTS.fcm import fts as fcm_fts
from pyFTS.partitioners import Grid
from pyFTS.common import Util

df = pd.read_csv('https://query.data.world/s/56i2vkijbvxhtv5gagn7ggk3zw3ksi', sep=';')

data = df['glo_avg'].values[:]


train = data[:7000]
test = data[7000:7500]

fs = Grid.GridPartitioner(data=train, npart=5)

model = fcm_fts.FCM_FTS(partitioner=fs, order=2, activation_function = Activations.relu)


model.fit(train, method='GD', alpha=0.02, momentum=0.8, iteractions=3 )

Util.persist_obj(model, 'fcm_fts10c')
'''
model = Util.load_obj('fcm_fts05c')
'''
#forecasts = model.predict(test)

#print(model)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[15,5])

ax.plot(test,label='Original')

forecasts = model.predict(test)

for w in np.arange(model.order):
  forecasts.insert(0,None)

ax.plot(forecasts, label=model.shortname)

plt.show()

print("")