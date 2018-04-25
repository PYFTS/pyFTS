#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import numpy as np
import matplotlib.pylab as plt
#from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
from pyFTS.common import Transformations

tdiff = Transformations.Differential(1)

from pyFTS.data import TAIEX

dataset = TAIEX.get_data()
'''
from pyFTS.partitioners import Grid, Util as pUtil
partitioner = Grid.GridPartitioner(data=dataset[:800], npart=10) #, transformation=tdiff)
'''
from pyFTS.benchmarks import benchmarks as bchmk, Util as bUtil, Measures, knn, quantreg, arima


from pyFTS.models import pwfts, song, ifts
from pyFTS.models.ensemble import ensemble

'''
model = knn.KNearestNeighbors("")
model.fit(dataset[:800])
Measures.get_distribution_statistics(dataset[800:1000], model)
#tmp = model.predict(dataset[800:1000], type='distribution')
#for tmp2 in tmp:
#    print(tmp2)
'''


'''
from pyFTS.partitioners import Grid, Util as pUtil
partitioner = Grid.GridPartitioner(data=dataset[:800], npart=10, transformation=tdiff)

model = pwfts.ProbabilisticWeightedFTS('',partitioner=partitioner)
model.append_transformation(tdiff)
model.fit(dataset[:800])
print(Measures.get_distribution_statistics(dataset[800:1000], model, steps_ahead=7))
#tmp = model.predict(dataset[800:1000], type='distribution', steps_ahead=7)
#for tmp2 in tmp:
#    print(tmp2)
'''

#'''

from pyFTS.benchmarks import arima, naive, quantreg

bchmk.sliding_window_benchmarks(dataset[:1000], 1000, train=0.8, inc=0.2,
                                #methods=[pwfts.ProbabilisticWeightedFTS],
                                benchmark_models=[],
                                benchmark_methods=[arima.ARIMA for k in range(4)]
                                    + [quantreg.QuantileRegression for k in range(2)]
                                    + [knn.KNearestNeighbors],
                                benchmark_methods_parameters=[
                                    {'order': (1, 0, 0)},
                                    {'order': (1, 0, 1)},
                                    {'order': (2, 0, 1)},
                                    {'order': (2, 0, 2)},
                                    {'order': 1, 'dist': True},
                                    {'order': 2, 'dist': True},
                                    {}
                                ],
                                #transformations=[tdiff],
                                orders=[1],
                                partitions=np.arange(30, 80, 5),
                                progress=False, type='distribution',
                                #steps_ahead=[1,4,7,10], #steps_ahead=[1]
                                #distributed=True, nodes=['192.168.0.110', '192.168.0.107','192.168.0.106'],
                                file="benchmarks.tmp", dataset="TAIEX", tag="comparisons")


#'''
'''
dat = pd.read_csv('pwfts_taiex_partitioning.csv', sep=';')
print(bUtil.analytic_tabular_dataframe(dat))
#print(dat["Size"].values[0])
'''
'''
train_split = 2000
test_length = 200

from pyFTS.partitioners import Grid, Util as pUtil
partitioner = Grid.GridPartitioner(data=dataset[:train_split], npart=30)
#partitioner = Grid.GridPartitioner(data=dataset[:train_split], npart=10, transformation=tdiff)

from pyFTS.common import fts,tree
from pyFTS.models import hofts, pwfts

pfts1_taiex = pwfts.ProbabilisticWeightedFTS("1", partitioner=partitioner)
#pfts1_taiex.append_transformation(tdiff)
pfts1_taiex.fit(dataset[:train_split], save_model=True, file_path='pwfts')
pfts1_taiex.shortname = "1st Order"

print(pfts1_taiex)

tmp = pfts1_taiex.predict(dataset[train_split:train_split+200], type='point',
                          method='heuristic')


print(tmp)

tmp = pfts1_taiex.predict(dataset[train_split:train_split+200], type='point',
                          method='expected_value')


print(tmp)
'''

'''
tmp = pfts1_taiex.predict(dataset[train_split:train_split+200], type='diPedro Pazzini
stribution', steps_ahead=20)


f, ax = plt.subplots(3, 4, figsize=[20,15])
tmp[0].plot(ax[0][0], title='t=1')
tmp[2].plot(ax[0][1], title='t=20')
tmp[4].plot(ax[0][2], title='t=40')
tmp[6].plot(ax[0][3], title='t=60')
tmp[8].plot(ax[1][0], title='t=80')
tmp[10].plot(ax[1][1], title='t=100')
tmp[12].plot(ax[1][2], title='t=120')
tmp[14].plot(ax[1][3], title='t=140')
tmp[16].plot(ax[2][0], title='t=160')
tmp[18].plot(ax[2][1], title='t=180')
tmp[20].plot(ax[2][2], title='t=200')


f, ax = plt.subplots(1, 1, figsize=[20,15])
bchmk.plot_distribution(ax, 'blue', tmp, f, 0, reference_data=dataset[train_split:train_split+200])

'''
