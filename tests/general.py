#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
from pyFTS.partitioners import Grid, Entropy, FCM, Huarng
from pyFTS.common import FLR,FuzzySet,Membership,Transformations
from pyFTS import fts,hofts,ifts,pwfts,tree, chen, pfts
from pyFTS.benchmarks import benchmarks as bchmk
from pyFTS.benchmarks import naive
from pyFTS.benchmarks import Measures
from numpy import random

#print(FCM.FCMPartitionerTrimf.__module__)

#gauss = random.normal(0,1.0,2000)
#gauss_teste = random.normal(0,1.0,400)


os.chdir("/home/petronio/dados/Dropbox/Doutorado/Disciplinas/AdvancedFuzzyTimeSeriesModels/")

#taiexpd = pd.read_csv("DataSets/TAIEX.csv", sep=",")
#taiex = np.array(taiexpd["avg"][:5000])

taiex = pd.read_csv("DataSets/TAIEX.csv", sep=",")
taiex_treino = np.array(taiex["avg"][2500:3900])
taiex_teste = np.array(taiex["avg"][3901:4500])

#print(len(taiex))

#from pyFTS.common import Util

#, ,

diff = Transformations.Differential(1)

#bchmk.sliding_window(taiex,2000,train=0.8, #transformation=diff, #models=[pwfts.ProbabilisticWeightedFTS],
#                     partitioners=[Grid.GridPartitioner, FCM.FCMPartitioner, Entropy.EntropyPartitioner],
#                     partitions=[10, 15, 20, 25, 30, 35, 40], dump=True, save=True, file="experiments/points.csv")


bchmk.allPointForecasters(taiex_treino, taiex_treino, 7, transformation=diff,
                          models=[ naive.Naive, pfts.ProbabilisticFTS, pwfts.ProbabilisticWeightedFTS],
                         statistics=True, residuals=False, series=False)

data_train_fs = Grid.GridPartitioner(taiex_treino, 10, transformation=diff).sets

fts1 = pfts.ProbabilisticFTS("")
fts1.appendTransformation(diff)
fts1.train(taiex_treino, data_train_fs, order=1)

print(fts1.forecast([5000, 5000]))

fts2 = pwfts.ProbabilisticWeightedFTS("")
fts2.appendTransformation(diff)
fts2.train(taiex_treino, data_train_fs, order=1)

print(fts2.forecast([5000, 5000]))


#tmp = Grid.GridPartitioner(taiex_treino,7,transformation=diff)

#for s in tmp.sets: print(s)