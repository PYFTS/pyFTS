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
from pyFTS import fts,hofts,ifts,pwfts,tree, chen, ensemble, song, yu, cheng, ismailefendi, sadaei, hwang
from pyFTS.benchmarks import naive, arima
from numpy import random
from pyFTS.benchmarks import benchmarks as bchmk
from pyFTS.benchmarks import arima, quantreg, Measures

os.chdir("/home/petronio/dados/Dropbox/Doutorado/Codigos/")

diff = Transformations.Differential(1)

passengers = pd.read_csv("DataSets/AirPassengers.csv", sep=",")
passengers = np.array(passengers["Passengers"])


e = ensemble.AllMethodEnsembleFTS()

fo_methods = [song.ConventionalFTS, chen.ConventionalFTS, yu.WeightedFTS, cheng.TrendWeightedFTS, sadaei.ExponentialyWeightedFTS,
              ismailefendi.ImprovedWeightedFTS]

ho_methods = [hofts.HighOrderFTS, hwang.HighOrderFTS]

fs = Grid.GridPartitioner(passengers, 10, transformation=diff)

e.appendTransformation(diff)

e.train(passengers, fs.sets, order=3)

"""

for method in fo_methods:
    model = method("")
    model.appendTransformation(diff)
    model.train(passengers, fs.sets)
    e.appendModel(model)


for method in ho_methods:
    for order in [1,2,3]:
        model = method("")
        model.appendTransformation(diff)
        model.train(passengers, fs.sets, order=order)
        e.appendModel(model)


arima100 = arima.ARIMA("", alpha=0.25)
#tmp.appendTransformation(diff)
arima100.train(passengers, None, order=(1,0,0))

arima101 = arima.ARIMA("", alpha=0.25)
#tmp.appendTransformation(diff)
arima101.train(passengers, None, order=(1,0,1))

arima200 = arima.ARIMA("", alpha=0.25)
#tmp.appendTransformation(diff)
arima200.train(passengers, None, order=(2,0,0))

arima201 = arima.ARIMA("", alpha=0.25)
#tmp.appendTransformation(diff)
arima201.train(passengers, None, order=(2,0,1))


e.appendModel(arima100)
e.appendModel(arima101)
e.appendModel(arima200)
e.appendModel(arima201)

e.train(passengers, None)


_mean = e.forecast(passengers, method="mean")
print(_mean)

_median = e.forecast(passengers, method="median")
print(_median)
"""
"""
_extremum = e.forecastInterval(passengers, method="extremum")
print(_extremum)

_quantile = e.forecastInterval(passengers, method="quantile", alpha=0.25)
print(_quantile)


_normal = e.forecastInterval(passengers, method="normal", alpha=0.25)
print(_normal)
"""

#"""
#_extremum = e.forecastAheadInterval(passengers, 10, method="extremum")
#print(_extremum)

#_quantile = e.forecastAheadInterval(passengers[:50], 40, method="quantile", alpha=0.25)
#print(_quantile)


#_normal = e.forecastAheadInterval(passengers, 10, method="normal", alpha=0.25)
#print(_normal)
#"""

#dist = e.forecastAheadDistribution(passengers, 20)

#print(dist)

bchmk.plot_compared_intervals_ahead(passengers[:120],[e], ['blue','red'],
                                    distributions=[True,False],  save=True, file="pictures/distribution_ahead_arma",
                                    time_from=60, time_to=10, tam=[12,5])



