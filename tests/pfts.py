#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
from pyFTS.partitioners import Grid
from pyFTS.common import FLR,FuzzySet,Membership,Transformations
from pyFTS import fts,hofts,ifts,pfts,tree, chen
from pyFTS.benchmarks import benchmarks as bchmk
from pyFTS.benchmarks import Measures
from numpy import random

gauss_treino = random.normal(0,1.0,1600)
gauss_teste = random.normal(0,1.0,400)


os.chdir("/home/petronio/dados/Dropbox/Doutorado/Disciplinas/AdvancedFuzzyTimeSeriesModels/")

#enrollments = pd.read_csv("DataSets/Enrollments.csv", sep=";")
#enrollments = np.array(enrollments["Enrollments"])

#taiex = pd.read_csv("DataSets/TAIEX.csv", sep=",")
#taiex_treino = np.array(taiex["avg"][2500:3900])
#taiex_teste = np.array(taiex["avg"][3901:4500])

#nasdaq = pd.read_csv("DataSets/NASDAQ_IXIC.csv", sep=",")
#nasdaq_treino = np.array(nasdaq["avg"][0:1600])
#nasdaq_teste = np.array(nasdaq["avg"][1601:2000])

diff = Transformations.Differential(1)

fs = Grid.GridPartitionerTrimf(gauss_treino,10)

#tmp = chen.ConventionalFTS("")

pfts1 = pfts.ProbabilisticFTS("1")
#pfts1.appendTransformation(diff)
pfts1.train(gauss_treino,fs,2)
#pfts2 = pfts.ProbabilisticFTS("n = 2")
#pfts2.appendTransformation(diff)
#pfts2.train(gauss_treino,fs,2)

#pfts3 = pfts.ProbabilisticFTS("n = 3")
#pfts3.appendTransformation(diff)
#pfts3.train(gauss_treino,fs,3)

#densities1 = pfts1.forecastAheadDistribution(gauss_teste[:50],2,1.50, parameters=2)

#print(bchmk.getDistributionStatistics(gauss_teste[:50], [pfts1,pfts2,pfts3], 20, 1.50))

sim_fs, sim_uod  =pfts1.generate_data(bins=10)

#print(pdf_fs)

#print(sim_fs)

print(sim_uod)

print(np.ravel(sim_uod))

#print(sim_uod)

#print (Measures.pdf(gauss_treino,bins=10))





