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

from pyFTS.data import TAIEX

taiex = TAIEX.get_data()

train = taiex[:3000]
test = taiex[3000:3200]

from pyFTS.common import Transformations
tdiff = Transformations.Differential(1)

'''
from pyFTS.partitioners import Grid, Util as pUtil
from pyFTS.common import FLR,FuzzySet,Membership,SortedCollection
taiex_fs1 = Grid.GridPartitioner(data=train, npart=30)
taiex_fs2 = Grid.GridPartitioner(data=train, npart=10, transformation=tdiff)

#pUtil.plot_partitioners(train, [taiex_fs1,taiex_fs2], tam=[15,7])

from pyFTS.common import fts,tree
from pyFTS.models import hofts, pwfts

pfts1_taiex = pwfts.ProbabilisticWeightedFTS("1", partitioner=taiex_fs1)
#pfts1_taiex.appendTransformation(diff)
pfts1_taiex.fit(train, save_model=True, file_path='pwfts')
pfts1_taiex.shortname = "1st Order"
print(pfts1_taiex)

'''

model = Util.load_obj('pwfts')

model.predict(test, type='distribution')
#'''