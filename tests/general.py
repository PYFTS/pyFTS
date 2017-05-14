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

"""
DATASETS
"""

#gauss = random.normal(0,1.0,5000)
#gauss_teste = random.normal(0,1.0,400)

taiexpd = pd.read_csv("DataSets/TAIEX.csv", sep=",")
taiex = np.array(taiexpd["avg"][:5000])

#nasdaqpd = pd.read_csv("DataSets/NASDAQ_IXIC.csv", sep=",")
#nasdaq = np.array(nasdaqpd["avg"][0:5000])

#sp500pd = pd.read_csv("DataSets/S&P500.csv", sep=",")
#sp500 = np.array(sp500pd["Avg"][11000:])
#del(sp500pd)

#sondapd = pd.read_csv("DataSets/SONDA_BSB_HOURLY_AVG.csv", sep=";")
#sondapd = sondapd.dropna(axis=0, how='any')
#sonda = np.array(sondapd["ws_10m"])
#del(sondapd)

#bestpd = pd.read_csv("DataSets/BEST_TAVG.csv", sep=";")
#best = np.array(bestpd["Anomaly"])

#print(lag)
#print(a)

#from pyFTS.benchmarks import benchmarks as bchmk
from pyFTS.benchmarks import distributed_benchmarks as bchmk
#from pyFTS.benchmarks import parallel_benchmarks as bchmk
from pyFTS.benchmarks import Util
from pyFTS.benchmarks import arima, quantreg

#Util.cast_dataframe_to_synthetic_point("experiments/taiex_point_analitic.csv","experiments/taiex_point_sintetic.csv",11)

#Util.plot_dataframe_point("experiments/taiex_point_sintetic.csv","experiments/taiex_point_analitic.csv",11)

#tmp = arima.ARIMA("")
#tmp.train(taiex[:1600], None, order=(2,0,2))
#teste = tmp.forecastInterval(taiex[1600:1605])

#tmp = quan#treg.QuantileRegression("")
#tmp.train(taiex[:1600], None, order=2)
#teste = tmp.forecast(taiex[1600:1605])

#print(taiex[1600:1605])
#print(teste)

#bchmk.teste(taiex,['192.168.0.109', '192.168.0.101'])

from pyFTS import song, chen, yu, cheng

"""
bchmk.point_sliding_window(sonda, 9000, train=0.8, inc=0.4,#models=[yu.WeightedFTS], # #
                     partitioners=[Grid.GridPartitioner], #Entropy.EntropyPartitioner], # FCM.FCMPartitioner, ],
                     partitions= np.arange(10,200,step=10), #transformation=diff,
                     dump=True, save=True, file="experiments/sondaws_point_analytic.csv",
                     nodes=['192.168.0.103', '192.168.0.106', '192.168.0.108', '192.168.0.109']) #, depends=[hofts, ifts])

diff = Transformations.Differential(1)

bchmk.point_sliding_window(sonda, 9000, train=0.8, inc=0.4, #models=[yu.WeightedFTS], # #
                     partitioners=[Grid.GridPartitioner], #Entropy.EntropyPartitioner], # FCM.FCMPartitioner, ],
                     partitions= np.arange(3,20,step=2), #transformation=diff,
                     dump=True, save=True, file="experiments/sondaws_point_analytic_diff.csv",
                     nodes=['192.168.0.103', '192.168.0.106', '192.168.0.108', '192.168.0.109']) #, depends=[hofts, ifts])
#"""

from pyFTS.partitioners import Grid
from pyFTS import pwfts

diff = Transformations.Differential(1)

fs = Grid.GridPartitioner(taiex[:2000], 10, transformation=diff)

tmp = pwfts.ProbabilisticWeightedFTS("")

tmp.appendTransformation(diff)

tmp.train(taiex[:1600], fs.sets, order=1)

x = tmp.forecastInterval(taiex[1600:1610])

print(taiex[1600:1610])
print(x)
