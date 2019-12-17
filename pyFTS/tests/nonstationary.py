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

from pyFTS.benchmarks import naive
from pyFTS.models.incremental import TimeVariant, IncrementalEnsemble
from pyFTS.models.nonstationary import common, perturbation, partitioners as nspart
from pyFTS.models.nonstationary import nsfts, util as nsUtil
from pyFTS.partitioners import Grid
from pyFTS.models import hofts, pwfts
from pyFTS.benchmarks import Measures
from pyFTS.common import Transformations, Util as cUtil

diff = Transformations.Differential(lag=1)

train = dataset[:1000]
test = dataset[1000:]

#grid = Grid.GridPartitioner(data=train, transformation=diff)
#model = pwfts.ProbabilisticWeightedFTS(partitioner=grid)
#model.append_transformation(diff)

model = naive.Naive()

model.fit(train)

for ct, ttrain, ttest in cUtil.sliding_window(test, 1000, .95, inc=.5):
  if model.shortname not in ('PWFTS','Naive'):
    model.predict(ttrain)
  print(ttest)
  if len(ttest) > 0:
    forecasts = model.predict(ttest, steps_ahead=10)
    measures = Measures.get_point_ahead_statistics(ttest[1:11], forecasts)

  print(measures)

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