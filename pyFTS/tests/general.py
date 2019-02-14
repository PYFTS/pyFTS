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
from pyFTS.models import chen, yu, cheng, ismailefendi, hofts, pwfts
from pyFTS.common import Transformations, Membership

tdiff = Transformations.Differential(1)


from pyFTS.data import TAIEX, SP500, NASDAQ, Malaysia, Enrollments

#from pyFTS.data import mackey_glass
#y = mackey_glass.get_data()

from pyFTS.partitioners import Grid
from pyFTS.models import pwfts, tsaur

x = [k for k in np.arange(-2*np.pi, 2*np.pi, 0.5)]
y = [np.sin(k) for k in x]

part = Grid.GridPartitioner(data=y, npart=35)
model = hofts.HighOrderFTS(order=2, partitioner=part)
model.fit(y)
forecasts = model.predict(y)

print([round(k,2) for k in y[2:]])
print([round(k,2) for k in forecasts[:-1]])