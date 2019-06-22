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

from pyFTS.models.multivariate import common, variable, mvfts, wmvfts
from pyFTS.models.seasonal import partitioner as seasonal
from pyFTS.models.seasonal.common import DateTime
from pyFTS.common import Membership

from pyFTS.data import SONDA

data = [k for k in SONDA.get_data('ws_10m') if k > 0.1 and k != np.nan and k is not None]
data = [np.nanmean(data[k:k+60]) for k in np.arange(0,len(data),60)]

train = data[:9000]
test = data[9000:10000]

fs = Grid.GridPartitioner(data=train, npart=95)

model = pwfts.ProbabilisticWeightedFTS(partitioner=fs, order=3)

model.fit(train)

model.predict(test)