#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import numpy as np
import matplotlib.pylab as plt
#from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
from pyFTS.common import Transformations

from pyFTS.data import TAIEX

dataset = TAIEX.get_data()

from pyFTS.benchmarks import benchmarks as bchmk

from pyFTS.models import pwfts

'''
bchmk.sliding_window_benchmarks(dataset, 1000, train=0.8, inc=0.2, methods=[pwfts.ProbabilisticWeightedFTS],
                                benchmark_models=False, orders=[1], partitions=[10], #np.arange(10,100,2),
                                progress=False, type='distribution',
                                distributed=False, nodes=['192.168.0.106', '192.168.0.105', '192.168.0.110'],
                                save=True, file="pwfts_taiex_interval.csv")
'''

train_split = 2000
test_length = 200

from pyFTS.partitioners import Grid, Util as pUtil
partitioner = Grid.GridPartitioner(data=dataset[:train_split], npart=30)

from pyFTS.common import fts,tree
from pyFTS.models import hofts, pwfts

pfts1_taiex = pwfts.ProbabilisticWeightedFTS("1", partitioner=partitioner)
pfts1_taiex.fit(dataset[:train_split], save_model=True, file_path='pwfts')
pfts1_taiex.shortname = "1st Order"

print(pfts1_taiex)

tmp = pfts1_taiex.predict(dataset[train_split:train_split+200], type='distribution', steps_ahead=20)
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