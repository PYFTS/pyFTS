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

train = dataset[:1000]
test = dataset[1000:]

from pyFTS.models.nonstationary import partitioners as nspart, nsfts, honsfts
fs = nspart.simplenonstationary_gridpartitioner_builder(data=train,npart=35,transformation=None)
print(fs)
model = honsfts.HighOrderNonStationaryFTS(partitioner=fs, order=2)
#model = nsfts.NonStationaryFTS(partitioner=fs)
model.fit(train)
forecasts = model.predict(test)
print(forecasts)