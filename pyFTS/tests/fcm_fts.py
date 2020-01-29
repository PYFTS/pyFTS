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

fs = Grid.GridPartitioner(data=train, npart=7)

model = fcm_fts.FCM_FTS(partitioner=fs, order=2, activation_function = Activations.relu)

model.fit(train,
          ngen=30, #number of generations
          mgen=7, # stop after mgen generations without improvement
          npop=10, # number of individuals on population
        pcruz=.5, # crossover percentual of population
        pmut=.3, # mutation percentual of population
        window_size = 7000,
        train_rate = .8,
        increment_rate =.2,
        experiments=1
        )

Util.persist_obj(model, 'fcm_fts10c')
'''
model = Util.load_obj('fcm_fts05c')
'''

forecasts = model.predict(test)

print(model)