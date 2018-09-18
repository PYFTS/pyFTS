#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import numpy as np
import matplotlib.pylab as plt
#from mpl_toolkits.mplot3d import Axes3D

import pandas as pd

from pyFTS.common import Util as cUtil, FuzzySet
from pyFTS.common import Transformations

tdiff = Transformations.Differential(1)

from pyFTS.data import TAIEX, SP500, NASDAQ

dataset = TAIEX.get_data()
#dataset = SP500.get_data()[11500:16000]
#dataset = NASDAQ.get_data()
#print(len(dataset))

from pyFTS.partitioners import Grid, Util as pUtil
partitioner = Grid.GridPartitioner(data=dataset[:2000], npart=20) #, transformation=tdiff)

print(partitioner)

#print(FuzzySet.__binary_search(7000, partitioner.sets, partitioner.ordered_sets))

print(FuzzySet.fuzzyfy([5000, 7000, 8000], partitioner, mode='vector', method='fuzzy', alpha_cut=.5))
print(FuzzySet.fuzzyfy([5000, 7000, 8000], partitioner, mode='sets', method='fuzzy', alpha_cut=.5))


"""

from pyFTS.benchmarks import benchmarks as bchmk, Util as bUtil, Measures, knn, quantreg, arima, naive

from pyFTS.models import pwfts, song, chen, ifts, hofts
from pyFTS.models.ensemble import ensemble

print(partitioner)

#model = chen.ConventionalFTS(partitioner=partitioner)
model = hofts.HighOrderFTS(partitioner=partitioner,order=2)
#model.append_transformation(tdiff)
model.fit(dataset[:2000])

print(model)

print(model.predict([3500, 7000], steps_ahead=5))



#cUtil.plot_rules(model, size=[20,20], rules_by_axis=5, columns=1)

#

print("fim")



'''
model = knn.KNearestNeighbors(order=3)
#model = ensemble.AllMethodEnsembleFTS("", partitioner=partitioner)
#model = arima.ARIMA("", order=(2,0,2))
#model = quantreg.QuantileRegression("", order=2, dist=True)
#model.append_transformation(tdiff)
model.fit(dataset[:800])
print(Measures.get_distribution_statistics(dataset[800:1000], model))
#tmp = model.predict(dataset[800:1000], type='distribution')
#for tmp2 in tmp:
#    print(tmp2)
#'''
'''

bchmk.sliding_window_benchmarks(dataset, 1000, train=0.8, inc=0.2,
                                methods=[hofts.HighOrderFTS], #[pwfts.ProbabilisticWeightedFTS],
                                benchmark_models=False,
                                transformations=[None],
                                orders=[1, 2, 3],
                                partitions=np.arange(30, 80, 5),
                                progress=False, type="point",
                                #steps_ahead=[1,2,4,6,8,10],
                                distributed=True, nodes=['192.168.0.110', '192.168.0.107', '192.168.0.106'],
                                file="benchmarks.db", dataset="TAIEX", tag="comparisons")



bchmk.sliding_window_benchmarks(dataset, 1000, train=0.8, inc=0.2,
                                methods=[hofts.HighOrderFTS],  # [pwfts.ProbabilisticWeightedFTS],
                                benchmark_models=False,
                                transformations=[tdiff],
                                orders=[1, 2, 3],
                                partitions=np.arange(3, 25, 2),
                                progress=False, type="point",
                                #steps_ahead=[1,2,4,6,8,10],
                                distributed=True, nodes=['192.168.0.110', '192.168.0.107', '192.168.0.106'],
                                file="benchmarks.db", dataset="NASDAQ", tag="comparisons")

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
'''

types = ['point','interval','distribution']
benchmark_methods=[[arima.ARIMA for k in range(8)] + [quantreg.QuantileRegression for k in range(4)]]
benchmark_methods=[
    [arima.ARIMA for k in range(4)] + [naive.Naive],
    [arima.ARIMA for k in range(8)] + [quantreg.QuantileRegression for k in range(4)],
    [arima.ARIMA for k in range(4)] + [quantreg.QuantileRegression for k in range(2)]
    + [knn.KNearestNeighbors for k in range(3)]
    ]
benchmark_methods_parameters= [
    [
        {'order': (1, 0, 0), 'alpha': .05},
        {'order': (1, 0, 0), 'alpha': .25},
        {'order': (1, 0, 1), 'alpha': .05},
        {'order': (1, 0, 1), 'alpha': .25},
        {'order': (2, 0, 1), 'alpha': .05},
        {'order': (2, 0, 1), 'alpha': .25},
        {'order': (2, 0, 2), 'alpha': .05},
        {'order': (2, 0, 2), 'alpha': .25},
        {'order': 1, 'alpha': .05},
        {'order': 1, 'alpha': .25},
        {'order': 2, 'alpha': .05},
        {'order': 2, 'alpha': .25}
    ]
]
benchmark_methods_parameters= [
    [
        {'order': (1, 0, 0)},
        {'order': (1, 0, 1)},
        {'order': (2, 0, 1)},
        {'order': (2, 0, 2)},
        {},
    ],[
        {'order': (1, 0, 0), 'alpha': .05},
        {'order': (1, 0, 0), 'alpha': .25},
        {'order': (1, 0, 1), 'alpha': .05},
        {'order': (1, 0, 1), 'alpha': .25},
        {'order': (2, 0, 1), 'alpha': .05},
        {'order': (2, 0, 1), 'alpha': .25},
        {'order': (2, 0, 2), 'alpha': .05},
        {'order': (2, 0, 2), 'alpha': .25},
        {'order': 1, 'alpha': .05},
        {'order': 1, 'alpha': .25},
        {'order': 2, 'alpha': .05},
        {'order': 2, 'alpha': .25}
    ],[
        {'order': (1, 0, 0)},
        {'order': (1, 0, 1)},
        {'order': (2, 0, 1)},
        {'order': (2, 0, 2)},
        {'order': 1, 'dist': True},
        {'order': 2, 'dist': True},
        {'order': 1}, {'order': 2}, {'order': 3},
    ]
]
dataset_name = "SP500"
tag = "ahead2"

from pyFTS.benchmarks import arima, naive, quantreg

for ct, type in enumerate(types):


    bchmk.sliding_window_benchmarks(dataset, 1000, train=0.8, inc=0.2,
                                    methods=[pwfts.ProbabilisticWeightedFTS],
                                    benchmark_models=False,
                                    #benchmark_methods=benchmark_methods[ct],
                                    #benchmark_methods_parameters=benchmark_methods_parameters[ct],
                                    transformations=[tdiff],
                                    orders=[1], #, 2, 3],
                                    partitions=[5], #np.arange(3, 35, 2),
                                    progress=False, type=type,
                                    steps_ahead=[2, 4, 6, 8, 10],
                                    distributed=True, nodes=['192.168.0.110', '192.168.0.107', '192.168.0.106'],
                                    file="benchmarks.db", dataset=dataset_name, tag=tag)
    bchmk.sliding_window_benchmarks(dataset, 1000, train=0.8, inc=0.2,
                                    methods=[pwfts.ProbabilisticWeightedFTS],
                                    benchmark_models=False,
                                    #benchmark_methods=benchmark_methods[ct],
                                    #benchmark_methods_parameters=benchmark_methods_parameters[ct],
                                    transformations=[None],
                                    orders=[1], #,2,3],
                                    partitions=[30], #np.arange(15, 85, 5),
                                    progress=False, type=type,
                                    steps_ahead=[2, 4, 6, 8, 10],
                                    distributed=True, nodes=['192.168.0.110', '192.168.0.107','192.168.0.106'],
                                    file="benchmarks.db", dataset=dataset_name, tag=tag)


'''
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
"""