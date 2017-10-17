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
from pyFTS import fts,hofts,ifts,pwfts,tree, chen
from pyFTS.benchmarks import benchmarks as bchmk
from pyFTS.benchmarks import Measures
from numpy import random

#gauss_treino = random.normal(0,1.0,1600)
#gauss_teste = random.normal(0,1.0,400)


os.chdir("/home/petronio/dados/Dropbox/Doutorado/Codigos/")

'''
enrollments = pd.read_csv("DataSets/Enrollments.csv", sep=";")
enrollments = np.array(enrollments["Enrollments"])
'''

taiexpd = pd.read_csv("DataSets/TAIEX.csv", sep=",")
data = np.array(taiexpd["avg"][:5000])
del(taiexpd)


import importlib
import pandas as pd
from pyFTS.partitioners import Grid
from pyFTS.common import FLR, FuzzySet, Membership, SortedCollection
from pyFTS import fts
from pyFTS import hofts
from pyFTS import pwfts
from pyFTS import tree
from pyFTS.benchmarks import benchmarks as bchmk

#uod = [10162, 21271]

fs1 = Grid.GridPartitioner(data[:3000], 30)
#for s in enrollments_fs1.sets:
#    print(s) #.partition_function(uod, 100))

pfts1 = pwfts.ProbabilisticWeightedFTS("1", partitioner=fs1)
pfts1.train(data, None, 1)
pfts1.shortname = "1st Order"

#print(pfts1_enrollments)

#tmp = pfts1.forecast(data[3000:3020])

#tmp = pfts1.forecastInterval(data[3000:3020])

tmp = pfts1.forecastDistribution(data[3500])

p = 0
for b in tmp[0].bins:
    p += tmp[0].density(b)

print(p)

#tmp = pfts1.forecastAheadInterval(data[3000:3020],20)

#tmp = pfts1.forecastAheadDistribution(data[3000:3020],20, method=3, h=0.45, kernel="gaussian")
#print(tmp[0])

#print(tmp[0].quantile([0.05, 0.95]))

#pfts1_enrollments.AprioriPDF
#norm = pfts1_enrollments.global_frequency_count
#uod = pfts1.get_UoD()

#for k in sorted(pfts1_enrollments.flrgs.keys())
#    flrg = pfts1_enrollments.flrgs[k]
#    tmp = flrg.get_LHSprobability(15000, norm, uod, 100)
#    print(tmp) #flrg.partition_function(uod,100))

#print("MARGINAL VERIFICATION")
#for s in sorted(pfts1_enrollments.flrgs.keys()):
#    flrg = pfts1_enrollments.flrgs[s]
    #print(flrg.get_LHSprobability(15000, norm, uod, 100))
#    print(sum([flrg.get_LHSprobability(k, norm, uod, 100) for k in np.linspace(uod[0],uod[1],100)]))



'''
pfts2_enrollments = pwfts.ProbabilisticWeightedFTS("2")
pfts2_enrollments.dump = False
pfts2_enrollments.shortname = "2nd Order"
pfts2_enrollments.train(enrollments, enrollments_fs1, 2)
pfts3_enrollments = pwfts.ProbabilisticWeightedFTS("3")
pfts3_enrollments.dump = False
pfts3_enrollments.shortname = "3rd Order"
pfts3_enrollments.train(enrollments, enrollments_fs1, 3)

bchmk.plot_compared_series(enrollments,[pfts1_enrollments,pfts2_enrollments, pfts3_enrollments],
                           ["red","blue","green"], linewidth=2,
                         typeonlegend=True,save=False,file="pictures/pwfts_enrollments_interval.png",
                           tam=[20,7],points=False, intervals=True)
'''






