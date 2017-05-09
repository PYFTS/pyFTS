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
#from pyFTS.benchmarks import benchmarks as bchmk
from pyFTS.benchmarks import naive, arima
from pyFTS.benchmarks import Measures
from numpy import random

os.chdir("/home/petronio/dados/Dropbox/Doutorado/Codigos/")

#enrollments = pd.read_csv("DataSets/Enrollments.csv", sep=";")
#enrollments = np.array(enrollments["Enrollments"])

#from pyFTS import song

#enrollments_fs = Grid.GridPartitioner(enrollments, 10).sets

#model = song.ConventionalFTS('')
#model.train(enrollments,enrollments_fs)
#teste = model.forecast(enrollments)

#print(teste)


#print(FCM.FCMPartitionerTrimf.__module__)

#gauss = random.normal(0,1.0,5000)
#gauss_teste = random.normal(0,1.0,400)

taiexpd = pd.read_csv("DataSets/TAIEX.csv", sep=",")
taiex = np.array(taiexpd["avg"][:5000])

#nasdaqpd = pd.read_csv("DataSets/NASDAQ_IXIC.csv", sep=",")
#nasdaq = np.array(nasdaqpd["avg"][0:5000])

#from statsmodels.tsa.arima_model import ARIMA as stats_arima
from statsmodels.tsa.tsatools import lagmat

#tmp = np.arange(10)

#lag, a = lagmat(tmp, maxlag=2, trim="both", original='sep')

#print(lag)
#print(a)

from pyFTS.benchmarks import distributed_benchmarks as bchmk
#from pyFTS.benchmarks import parallel_benchmarks as bchmk
from pyFTS.benchmarks import Util
from pyFTS.benchmarks import arima, quantreg

#Util.cast_dataframe_to_sintetic_point("experiments/taiex_point_analitic.csv","experiments/taiex_point_sintetic.csv",11)

#Util.plot_dataframe_point("experiments/taiex_point_sintetic.csv","experiments/taiex_point_analitic.csv",11)

#tmp = arima.ARIMA("")
#tmp.train(taiex[:1600], None, order=(1,0,1))
#teste = tmp.forecast(taiex[1600:1610])

#tmp = quantreg.QuantileRegression("")
#tmp.train(taiex[:1600], None, order=1)
#teste = tmp.forecast(taiex[1600:1610])

#print(taiex[1600:1610])
#print(teste)

#bchmk.teste(taiex,['192.168.0.109', '192.168.0.101'])

bchmk.point_sliding_window(taiex,2000,train=0.8, #models=[yu.WeightedFTS], # #
                     partitioners=[Grid.GridPartitioner], #Entropy.EntropyPartitioner], # FCM.FCMPartitioner, ],
                     partitions= np.arange(10,200,step=10), #transformation=diff,
                     dump=True, save=True, file="experiments/taiex_point_analytic.csv",
                     nodes=['192.168.0.102', '192.168.0.109', '192.168.0.106']) #, depends=[hofts, ifts])

diff = Transformations.Differential(1)

bchmk.point_sliding_window(taiex,2000,train=0.8, #models=[yu.WeightedFTS], # #
                     partitioners=[Grid.GridPartitioner], #Entropy.EntropyPartitioner], # FCM.FCMPartitioner, ],
                     partitions= np.arange(10,200,step=10), transformation=diff,
                     dump=True, save=True, file="experiments/taiex_point_analytic_diff.csv",
                     nodes=['192.168.0.102', '192.168.0.109', '192.168.0.106']) #, depends=[hofts, ifts])

#bchmk.testa(taiex,[10,20],partitioners=[Grid.GridPartitioner], nodes=['192.168.0.109', '192.168.0.101'])

#parallel_util.explore_partitioners(taiex,20)

#nasdaqpd = pd.read_csv("DataSets/NASDAQ_IXIC.csv", sep=",")
#nasdaq = np.array(nasdaqpd["avg"][:5000])

#taiex = pd.read_csv("DataSets/TAIEX.csv", sep=",")
#taiex_treino = np.array(taiex["avg"][2500:3900])
#taiex_teste = np.array(taiex["avg"][3901:4500])

#print(len(taiex))

#from pyFTS.common import Util

#, ,

#diff = Transformations.Differential(1)


#bchmk.external_point_sliding_window([naive.Naive, arima.ARIMA, arima.ARIMA, arima.ARIMA, arima.ARIMA, arima.ARIMA, arima.ARIMA],
#                                    [None, (1,0,0),(1,1,0),(2,0,0), (2,1,0), (1,1,1), (1,0,1)],
#                                    gauss,2000,train=0.8, dump=True, save=True, file="experiments/arima_gauss.csv")


#bchmk.interval_sliding_window(gauss,2000,train=0.8, #transformation=diff, #models=[pwfts.ProbabilisticWeightedFTS], # #
#                     partitioners=[Grid.GridPartitioner], #Entropy.EntropyPartitioner], # FCM.FCMPartitioner, ],
#                     partitions= np.arange(10,200,step=5), #
#                     dump=True, save=False, file="experiments/nasdaq_interval.csv")

#3bchmk.ahead_sliding_window(taiex,2000,train=0.8, steps=20, resolution=250, #transformation=diff, #models=[pwfts.ProbabilisticWeightedFTS], # #
#                    partitioners=[Grid.GridPartitioner], #Entropy.EntropyPartitioner], # FCM.FCMPartitioner, ],
#                    partitions= np.arange(10,200,step=10), #
#                     dump=True, save=True, file="experiments/taiex_ahead.csv")


#bchmk.allPointForecasters(taiex_treino, taiex_treino, 95, #transformation=diff,
#                          models=[ naive.Naive, pfts.ProbabilisticFTS, pwfts.ProbabilisticWeightedFTS],
#                         statistics=True, residuals=False, series=False)

#data_train_fs = Grid.GridPartitioner(nasdaq[:1600], 95).sets

#fts1 = pwfts.ProbabilisticWeightedFTS("")
#fts1.appendTransformation(diff)
#fts1.train(nasdaq[:1600], data_train_fs, order=1)

#_crps1, _crps2, _t1, _t2 = bchmk.get_distribution_statistics(nasdaq[1600:2000], fts1, steps=20, resolution=200)

#print(_crps1, _crps2, _t1, _t2)

#print(fts1.forecast([5000, 5000]))

#fts2 = pwfts.ProbabilisticWeightedFTS("")
#fts2.appendTransformation(diff)
#fts2.train(taiex_treino, data_train_fs, order=1)

#print(fts2.forecast([5000, 5000]))


#tmp = Grid.GridPartitioner(taiex_treino,7,transformation=diff)

#for s in tmp.sets: print(s)