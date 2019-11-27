import os
import numpy as np
from pyFTS.common import Membership, Transformations
from pyFTS.models.nonstationary import common, perturbation, partitioners as nspart, util
from pyFTS.models.nonstationary import nsfts, cvfts
from pyFTS.partitioners import Grid, Entropy
import matplotlib.pyplot as plt
from pyFTS.common import Util as cUtil
import pandas as pd

from pyFTS.data import TAIEX, NASDAQ, SP500, artificial, mackey_glass

#mackey_glass.get_data()

dataset =  TAIEX.get_data()

train_split = 1000
test_length = 200

from pyFTS.common import Transformations


from pyFTS.partitioners import Grid, Util as pUtil
from pyFTS.benchmarks import benchmarks as bchmk
from pyFTS.models import chen, hofts, pwfts, hwang
from pyFTS.models.incremental import TimeVariant, IncrementalEnsemble

train = dataset[:1000]
test = dataset[1000:]

window = 100

batch = 10

num_models = 3

model1 = TimeVariant.Retrainer(partitioner_method=Grid.GridPartitioner, partitioner_params={'npart': 35},
                               fts_method=pwfts.ProbabilisticWeightedFTS, fts_params={}, order=1,
                               batch_size=batch, window_length=window * num_models)

model2 = IncrementalEnsemble.IncrementalEnsembleFTS(partitioner_method=Grid.GridPartitioner,
                                                    partitioner_params={'npart': 35},
                                                    fts_method=pwfts.ProbabilisticWeightedFTS, fts_params={}, order=1,
                                                    batch_size=int(batch / 3), window_length=window,
                                                    num_models=num_models)

model1.fit(train)
model2.fit(train)

print(len(test))
'''
forecasts1 = model1.predict(test[:-10])
print(len(forecasts1))
forecasts1 = model1.predict(test[-10:], steps_ahead=10)
print(len(forecasts1))
'''
forecasts2 = model2.predict(test[:-10])
print(len(forecasts2))
forecasts2 = model2.predict(test[-10:], steps_ahead=10)
print(len(forecasts2))


'''
from pyFTS.models.nonstationary import partitioners as nspart, nsfts, honsfts
fs = nspart.simplenonstationary_gridpartitioner_builder(data=train,npart=35,transformation=None)
print(fs)
#model = honsfts.HighOrderNonStationaryFTS(partitioner=fs, order=2)
model = nsfts.WeightedNonStationaryFTS(partitioner=fs)
model.fit(train)
print(model)
forecasts = model.predict(test)
#print(forecasts)
'''