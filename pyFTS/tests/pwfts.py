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

from pyFTS.benchmarks import benchmarks as bchmk, Measures
from pyFTS.models import pwfts,hofts,ifts
from pyFTS.partitioners import Grid, Util as pUtil

fs = Grid.GridPartitioner(data=train, npart=30) #, transformation=tdiff)

model1 = hofts.HighOrderFTS(partitioner=fs, lags=[1,2])#lags=[0,1])
model1.shortname = "1"
model2 = pwfts.ProbabilisticWeightedFTS(partitioner=fs, lags=[1,2])
#model2.append_transformation(tdiff)
model2.shortname = "2"
#model = pwfts.ProbabilisticWeightedFTS(partitioner=fs, order=2)# lags=[1,2])

model1.fit(train)
model2.fit(train)

#print(model1)

#print(model2)

for model in [model1, model2]:
    #forecasts = model.predict(test)
    print(model.shortname)
    print(Measures.get_point_statistics(test, model))

#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles, labels, loc=2, bbox_to_anchor=(1, 1))

#print(Measures.get_point_statistics(test,model))


'''
bchmk.sliding_window_benchmarks(train,1000,0.8,
                                methods=[pwfts.ProbabilisticWeightedFTS], #,ifts.IntervalFTS],
                                orders=[1,2,3],
                                partitions=[10])
'''
'''

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

