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
from pyFTS.common import FLR,FuzzySet,Membership
from pyFTS import fts
from pyFTS import hofts
from pyFTS import ifts
from pyFTS import pfts
from pyFTS import tree
from pyFTS.benchmarks import benchmarks as bchmk


os.chdir("/home/petronio/dados/Dropbox/Doutorado/Disciplinas/AdvancedFuzzyTimeSeriesModels/")

enrollments = pd.read_csv("DataSets/Enrollments.csv", sep=";")
enrollments = np.array(enrollments["Enrollments"])

enrollments_fs1 = Grid.GridPartitionerTrimf(enrollments,6)

pfts1_enrollments = pfts.ProbabilisticFTS("1")
pfts1_enrollments.train(enrollments,enrollments_fs1,1)
pfts1_enrollments.shortname = "1st Order"
pfts2_enrollments = pfts.ProbabilisticFTS("2")
pfts2_enrollments.dump = False
pfts2_enrollments.shortname = "2nd Order"
pfts2_enrollments.train(enrollments,enrollments_fs1,2)


pfts1_enrollments.forecastAheadDistribution2(enrollments[:15],5,100)
