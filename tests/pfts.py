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


os.chdir("/home/petronio/dados/Dropbox/Doutorado/Disciplinas/AdvancedFuzzyTimeSeriesModels/")

enrollments = pd.read_csv("DataSets/Enrollments.csv", sep=";")
enrollments = np.array(enrollments["Enrollments"])

#diff = Transformations.Differential(1)

fs = Grid.GridPartitionerTrimf(enrollments,6)

#tmp = chen.ConventionalFTS("")

pfts1 = pfts.ProbabilisticFTS("1")
#pfts1.appendTransformation(diff)
pfts1.train(enrollments,fs,1)

#bchmk.plotComparedIntervalsAhead(enrollments,[pfts1], ["blue"],[True],5,10)

pfts1.forecastAheadDistribution(enrollments,5,1, parameters=True)
