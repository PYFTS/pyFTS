#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import numpy as np
import matplotlib.pylab as plt
#from mpl_toolkits.mplot3d import Axes3D

import pandas as pd

from pyFTS.common import Util as cUtil, FuzzySet
from pyFTS.partitioners import Grid, Entropy, Util as pUtil, Simple
from pyFTS.benchmarks import benchmarks as bchmk, Measures
from pyFTS.models import chen, yu, cheng, ismailefendi, hofts, pwfts, tsaur, song, sadaei
from pyFTS.common import Transformations, Membership

from pyFTS.data import TAIEX

data = TAIEX.get_data()

fs = Grid.GridPartitioner(data=data, npart=23)

test = [2000, 5000, 5500, 12000]

for method in [yu.WeightedFTS, tsaur.MarkovWeightedFTS, song.ConventionalFTS, sadaei.ExponentialyWeightedFTS, ismailefendi.ImprovedWeightedFTS,
               chen.ConventionalFTS, cheng.TrendWeightedFTS, hofts.HighOrderFTS, pwfts.ProbabilisticWeightedFTS]:
    model = method(partitioner=fs)
    model.fit(data)
    print(model.forecast(test))

