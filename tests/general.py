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
from pyFTS import fts,hofts,ifts,pwfts,tree, chen
from pyFTS.benchmarks import benchmarks as bchmk
from pyFTS.benchmarks import naive, arima
from pyFTS.benchmarks import Measures
from numpy import random

#print(FCM.FCMPartitionerTrimf.__module__)

#gauss = random.normal(0,1.0,5000)
#gauss_teste = random.normal(0,1.0,400)


os.chdir("/home/petronio/dados/Dropbox/Doutorado/Disciplinas/AdvancedFuzzyTimeSeriesModels/")

#taiexpd = pd.read_csv("DataSets/TAIEX.csv", sep=",")
#taiex = np.array(taiexpd["avg"][:5000])

nasdaqpd = pd.read_csv("DataSets/NASDAQ_IXIC.csv", sep=",")
nasdaq = np.array(nasdaqpd["avg"][:5000])

#taiex = pd.read_csv("DataSets/TAIEX.csv", sep=",")
#taiex_treino = np.array(taiex["avg"][2500:3900])
#taiex_teste = np.array(taiex["avg"][3901:4500])

#print(len(taiex))

#from pyFTS.common import Util

#, ,

#diff = Transformations.Differential(1)


bchmk.external_point_sliding_window([naive.Naive, arima.ARIMA, arima.ARIMA, arima.ARIMA, arima.ARIMA, arima.ARIMA, arima.ARIMA],
                                    [None, (1,0,0),(1,1,0),(2,0,0), (2,1,0), (1,1,1), (1,0,1)],
                                    nasdaq,2000,train=0.8, #transformation=diff, #models=[pwfts.ProbabilisticWeightedFTS], # #
                                    dump=True, save=True, file="experiments/arima_nasdaq.csv")


#bchmk.point_sliding_window(taiex,2000,train=0.8, #transformation=diff, #models=[pwfts.ProbabilisticWeightedFTS], # #
#                     partitioners=[Grid.GridPartitioner], #Entropy.EntropyPartitioner], # FCM.FCMPartitioner, ],
#                     partitions= [45,55, 65, 75, 85, 95,105,115,125,135, 150], #np.arange(5,150,step=10), #
#                     dump=True, save=True, file="experiments/taiex_point_new.csv")


#bchmk.allPointForecasters(taiex_treino, taiex_treino, 95, #transformation=diff,
#                          models=[ naive.Naive, pfts.ProbabilisticFTS, pwfts.ProbabilisticWeightedFTS],
#                         statistics=True, residuals=False, series=False)

#data_train_fs = Grid.GridPartitioner(taiex_treino, 10, transformation=diff).sets

#fts1 = pfts.ProbabilisticFTS("")
#fts1.appendTransformation(diff)
#fts1.train(taiex_treino, data_train_fs, order=1)

#print(fts1.forecast([5000, 5000]))

#fts2 = pwfts.ProbabilisticWeightedFTS("")
#fts2.appendTransformation(diff)
#fts2.train(taiex_treino, data_train_fs, order=1)

#print(fts2.forecast([5000, 5000]))


#tmp = Grid.GridPartitioner(taiex_treino,7,transformation=diff)

#for s in tmp.sets: print(s)