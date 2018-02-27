#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
from pyFTS.common import Transformations

from pyFTS.benchmarks import benchmarks as bchmk

bc = Transformations.BoxCox(0)
diff = Transformations.Differential(1)
#ix = SeasonalIndexer.LinearSeasonalIndexer([12, 24], [720, 1],[False, False])

"""
DATASETS
"""

from pyFTS.data import Enrollments

data = Enrollments.get_data()

from pyFTS.partitioners import Grid
from pyFTS.models import song, chen, yu, sadaei, ismailefendi, cheng, hofts

train = data
test = data

fs = Grid.GridPartitioner(train, 10) #, transformation=bc)

#tmp = bchmk.simpleSearch_RMSE(train, test, hofts.HighOrderFTS, range(4,12), [2], tam=[10, 5])

model = hofts.HighOrderFTS("", partitioner=fs)
model.fit(train, order=3)

print(model)